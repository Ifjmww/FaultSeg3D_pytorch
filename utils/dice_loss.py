import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-5):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        # 将预测结果和目标标签转换为二进制形式
        pred = pred[:, 1, :, :]  # 取第二个通道的预测结果
        target = target.float()

        # 计算Dice系数的分子和分母
        intersection = (pred * target).sum()
        dice_coefficient = (2. * intersection + self.epsilon) / (pred.sum() + target.sum() + self.epsilon)

        # 计算Dice Loss
        loss = 1 - dice_coefficient
        return loss
