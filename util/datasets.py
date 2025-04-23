# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

import os
from PIL import Image
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)
    root = os.path.join(args.data_path, is_train)
    dataset = datasets.ImageFolder(root, transform=transform)

    return dataset


def build_transform(is_train, args):
    # mean = [0.5] * 3  # 使用适合灰度图像转换为3通道后的均值
    # std = [0.5] * 3   # 使用适合灰度图像转换为3通道后的标准差
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train == 'train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def build_multi_dataset(is_train, args):
    # 设置数据集路径
    data_subdir = 'train' if is_train == 'train' else 'test'
    dataset_path = os.path.join(args.data_path, data_subdir)

    # 构建图像增强和数据集
    transform = build_transform(is_train, args)
    dataset = CFP_OCT_Dataset(dataset_path, transform=transform)
    return dataset


class CFP_OCT_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_pairs = []
        self.class_to_idx = {}  # 增加class_to_idx属性

        # 遍历类别、患者ID、CFP和OCT子目录，构建数据对
        for idx, category in enumerate(os.listdir(root_dir)):
            category_path = os.path.join(root_dir, category)
            if os.path.isdir(category_path):
                # 将类别名称映射到整数标签
                self.class_to_idx[category] = idx
                for patient_id in os.listdir(category_path):
                    patient_path = os.path.join(category_path, patient_id)
                    cfp_path = os.path.join(patient_path, "CFP")
                    oct_path = os.path.join(patient_path, "OCT")

                    # 获取CFP图像
                    cfp_image = None
                    for cfp_file in os.listdir(cfp_path):
                        cfp_image = os.path.join(cfp_path, cfp_file)
                        break  # 每个患者的CFP目录中只需获取一张CFP图像

                    # 获取所有OCT图像
                    oct_images = [os.path.join(oct_path, oct_file) for oct_file in os.listdir(oct_path)]

                    # 存储 (CFP, OCT, category, patient_id) 数据对
                    if cfp_image and oct_images:
                        for oct_image in oct_images:
                            self.data_pairs.append((cfp_image, oct_image, self.class_to_idx[category], patient_id))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        cfp_path, oct_path, category, patient_id = self.data_pairs[idx]

        # 加载图像
        cfp_image = Image.open(cfp_path).convert("RGB")
        oct_image = Image.open(oct_path).convert("RGB")
        if self.transform:
            cfp_image = self.transform(cfp_image)
            oct_image = self.transform(oct_image)

        return cfp_image, oct_image, category, patient_id
