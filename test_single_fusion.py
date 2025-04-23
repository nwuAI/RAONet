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
    parser.add_argument('--model_wts', type=str,
                        default='',
                        help='checkpoint path')

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

    dataset_val = build_dataset(is_train='test', args=args)
    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    retfound = load_retfound_model('vit_large_patch16', args.retfound_checkpoint, num_classes=6)

    resnet = load_resnet_model(args.resnet_checkpoint, 6)

    fusion_model = FusionModel(
        retfound, resnet,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim
    ).to(args.device)

    criterion = torch.nn.CrossEntropyLoss()

    fusion_model.load_state_dict(args.model_wts)

    fusion_model.eval()
    auc_roc, acc, all_labels, pred_labels, _ = validate_model_auc_accuracy(fusion_model, val_loader,
                                                                                  args.device, criterion)

    print("acc:", acc)
    print("auc_roc:", auc_roc)


if __name__ == '__main__':
    main()
