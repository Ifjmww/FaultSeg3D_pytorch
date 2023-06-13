# 数据加载部分，transform在这里统一设置为None，数据增强在训练前单独完成，训练、验证、预测的数据都是经过(x-mean)/std正则化后的;
import numpy as np
import os
import torch
from torch.utils.data import Dataset


class FaultDataset(Dataset):
    """
    Load_Dataset
    """

    def __init__(self, path, mode='train', transform=None):
        self.path = path
        self.transform = transform
        self.mode = mode

        self.image_list, self.label_list = self.load_data()

    def __getitem__(self, index):
        image = np.load(self.image_list[index])
        if len(self.label_list) == 0:
            label = np.zeros(image.shape)
        else:
            label = np.load(self.label_list[index])

        img = image
        if len(img.shape) == 3:
            img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

        x = torch.from_numpy(img)
        y = torch.from_numpy(label)

        data = {'x': x.float(), 'y': y.float()}

        return data

    def __len__(self):
        return len(self.image_list)

    def load_data(self):
        """

        :return:
        """
        img_list = []
        label_list = []
        label_pred_list = []
        img_path = os.path.join(self.path, 'x/')
        label_path = os.path.join(self.path, 'y/')
        for item in os.listdir(img_path):
            img_list.append(os.path.join(img_path, item))
            # 由于x和y的文件名一样，所以用一步加载进来
            label_list.append(os.path.join(label_path, item))
        if self.mode != 'pred':
            return img_list, label_list
        else:
            return img_list, label_pred_list
