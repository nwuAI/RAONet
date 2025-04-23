import os
import copy

import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader
from util.datasets import build_dataset
from models.model_fusion import FusionModel
from util.misc import validate_model_auc_accuracy, load_retfound_model, load_resnet_model


def parse_args():
    parser = argparse.ArgumentParser(description='CFP or OCT train')

    # 路径参数
    parser.add_argument('--data_path', type=str,
                        default='./images/CFP/',
                        help='data_path')
    parser.add_argument('--save_dir', type=str,
                        default='./checkpoints_dir/multi/',
                        help='save_dir')
    parser.add_argument('--retfound_checkpoint', type=str,
                        default='',
                        help='REFound checkpoint path')
    parser.add_argument('--resnet_checkpoint', type=str,
                        default='',
                        help='ResNet checkpoint path')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch_size (default:32)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='train epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learn rate')

    # 模型参数
    parser.add_argument('--num_classes', type=int, default=5,
                        help='num_classes')
    parser.add_argument('--embed_dim', type=int, default=3072,
                        help='embed_dim')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--seed', default=0, type=int)

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')

    # 设备参数
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device')

    return parser.parse_args()


def main():
    # 解析参数
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 创建保存目录
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # 构建数据集
    dataset_train = build_dataset(is_train='train', args=args)
    dataset_val = build_dataset(is_train='test', args=args)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    retfound = load_retfound_model('vit_large_patch16', args.retfound_checkpoint, num_classes=6)

    resnet = load_resnet_model(args.resnet_checkpoint, 6)

    # 初始化模型
    fusion_model = FusionModel(
        retfound, resnet,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim
    ).to(args.device)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # 训练循环
    best_auc = 0.0
    best_acc = 0.0
    val_loss_count = 0
    prev_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        fusion_model.train()  # 确保模型处于训练模式
        running_loss = 0.0
        running_corrects = 0

        for inputs1, labels1 in train_loader:
            inputs1 = inputs1.to(args.device)
            labels = labels1.to(args.device)  # 假设两个加载器的标签是相同的

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = fusion_model(inputs1)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs1.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{args.num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

        # 验证阶段
        fusion_model.eval()  # 设置模型为评估模式
        auc_roc, acc, all_labels, pred_labels, val_loss = validate_model_auc_accuracy(fusion_model, val_loader,
                                                                                      args.device, criterion)

        # 如果当前epoch的验证AUC-ROC更高，保存模型
        if (acc > best_acc) or (acc == best_acc and auc_roc > best_auc):
            best_acc = acc
            best_auc = auc_roc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(fusion_model.state_dict())
            print(f'Saving with AUC-ROC of {auc_roc:.4f}, ACC of {acc:.4f}')

        if val_loss < prev_val_loss:
            val_loss_count = 0
        else:
            val_loss_count += 1

        if val_loss_count == 10:
            print("Early stopping triggered due to no improvement in validation loss.")
            break

        prev_val_loss = val_loss

    fusion_model.load_state_dict(best_model_wts)

    file_path = os.path.join(args.save_dir, f'bestckpt_{best_epoch}_{best_acc:.4f}_{best_auc:.4f}.pth')
    torch.save(fusion_model.state_dict(), file_path)
    print(f'Model saved to {args.save_dir}')


if __name__ == '__main__':
    main()
