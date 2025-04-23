# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Partly revised by YZ @UCL&Moorfields
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    带有全局平均池化支持的Vision Transformer模型。
    """
    def __init__(self, global_pool=False, **kwargs):
        """
        初始化函数
        """
        # 先调用父类的初始化函数完成父类初始化
        super(VisionTransformer, self).__init__(**kwargs)

        # 根据global_pool的值，决定是否启用全局平均池化
        self.global_pool = global_pool
        # 如果global_pool为True，创建一个新的归一化层，并删除原始归一化层
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # 删除原始的归一化层

    def forward_features(self, x):
        """
        模型的特征前向传播函数
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        # 使用 cls_token 扩展一个与B相同大小的张量，然后将其与x连接在一起
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        # 将位置嵌入 pos_embed 加到x上，并通过 pos_drop 进行 Dropout 处理
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # 通过迭代执行多个 blocks（Transformer 编码块），将x进行特征提取和变换
        for blk in self.blocks:
            x = blk(x)
        # print("x", x.shape)
        # 如果启用全局平均池化，则对x进行平均池化操作（不包括 cls_token），然后通过新的归一化层fc_norm进行归一化，并将结果赋值给 outcome
        if self.global_pool:
            x = x[:, 1:, :]
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            # print("x_global",x.shape)
            outcome = self.fc_norm(x)
        else:  # 如果未启用全局平均池化，则对x进行归一化操作，并将第一个特征向量x[:, 0]赋值给 outcome
            x = self.norm(x)
            outcome = x[:, 0]

        return x


def vit_large_patch16(**kwargs):
    """
    创建一个基于Vision Transformer的模型。
    patch_size表示输入图像被划分为的图像块(patch)的大小为16*16
    embed_dim表示嵌入维度为1024
    num_heads表示模型的深度，即Transformer编码器的层数
    mlp_ratio表示MLP(多层感知机)中隐藏层维度相对于嵌入维度的扩展倍数
    qkv_bias表示是否在查询、键、值的线性变换中应用偏置
    norm_layer这块选择选择归一化层的类型
    """
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == '__main__':
    # 创建随机输入数据，模拟32张3通道224x224的图像
    x = torch.randn((32, 3, 224, 224))

    # 使用 vit_large_patch16 函数创建 VisionTransformer 模型实例
    model = vit_large_patch16(global_pool=True)

    # 将模型的输出传递给测试输入
    out = model.forward_features(x)

    # 打印输出形状，确保输出维度正确
    print("Output shape:", out.shape)
