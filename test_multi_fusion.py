import torch
import argparse
from torch.utils.data import DataLoader
from util.datasets import build_multi_dataset
from models.multi_model_fusion import FusionModel
from util.misc import validate_multi_auc_accuracy, load_retfound_model, load_resnet_model


def parse_args():
    """整合所有命令行参数解析"""
    parser = argparse.ArgumentParser(description='CFP-OCT test')

    # 路径参数
    parser.add_argument('--data_path', type=str,
                        default='./images',
                        help='data_path')
    parser.add_argument('--save_dir', type=str,
                        default='./checkpoints_dir/multi/',
                        help='save_dir')
    parser.add_argument('--retfound_cfp_checkpoint', type=str,
                        default='./',
                        help='CFP REFound checkpoint path')
    parser.add_argument('--retfound_oct_checkpoint', type=str,
                        default='./',
                        help='OCT REFound checkpoint path')
    parser.add_argument('--resnet_cfp_checkpoint', type=str,
                        default='./',
                        help='CFP ResNet checkpoint path')
    parser.add_argument('--resnet_oct_checkpoint', type=str,
                        default='./',
                        help='OCT ResNetcheckpoint path')

    # 模型参数
    parser.add_argument('--num_classes', type=int, default=6,
                        help='num_classes')
    parser.add_argument('--embed_dim', type=int, default=6144,
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

    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device')

    return parser.parse_args()


def main():
    args = parse_args()

    dataset_train = build_multi_dataset(is_train='train', args=args)
    dataset_val = build_multi_dataset(is_train='test', args=args)

    val_loader = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)

    retfound_cfp = load_retfound_model('vit_large_patch16', args.retfound_cfp_checkpoint, num_classes=6)
    retfound_oct = load_retfound_model('vit_large_patch16', args.retfound_oct_checkpoint, num_classes=6)
    resnet_cfp = load_resnet_model(args.resnet_cfp_checkpoint, 6)
    resnet_oct = load_resnet_model(args.resnet_oct_checkpoint, 6)

    fusion_model = FusionModel(
        retfound_cfp, retfound_oct, resnet_cfp, resnet_oct,
        num_classes=6,
        embed_dim=6144
    ).to(args.device)

    criterion = torch.nn.CrossEntropyLoss()
    fusion_model.load_state_dict(args.model_wts)

    fusion_model.eval()

    auc_roc, acc, all_labels, pred_labels, _ = validate_multi_auc_accuracy(fusion_model, val_loader, args.device,
                                                                           args.num_classes, criterion)

    print("acc:", acc)
    print("auc_roc:", auc_roc)


if __name__ == '__main__':
    main()
