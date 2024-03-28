import os
import torch
from dataloader.dataloader import FaultDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils.dice_loss import DiceLoss


def save_args_info(args):
    # save args to config.txt
    argsDict = args.__dict__
    result_path = './EXP/' + '/' + args.exp + '/'

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if args.mode == 'train':
        with open(result_path + 'config.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
    elif args.mode == 'valid_only':
        with open(result_path + 'config_valid_only.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')
    elif args.mode == 'pred':
        with open(result_path + 'config_pred.txt', 'w') as f:
            f.writelines('------------------ start ------------------' + '\n')
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
            f.writelines('------------------- end -------------------')


def load_data(args):
    # args.mode=['train', 'valid_only', 'pred']
    if args.mode == 'train':
        # 训练时的训练集
        train_dataset = FaultDataset(args.train_path, args.mode, transform=None)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)

        valid_dataset = FaultDataset(args.valid_path, args.mode, transform=None)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size_not_train, shuffle=True, num_workers=args.workers, drop_last=True)

        print("--- create train dataloader ---")
        print(len(train_dataset), ", train dataset created")
        print(len(train_dataloader), ", train dataloader created")

        print("--- create valid dataloader ---")
        print(len(valid_dataset), ", valid dataset created")
        print(len(valid_dataloader), ", valid dataloaders created")

        return train_dataloader, valid_dataloader

    elif args.mode == 'valid_only':
        dataset = FaultDataset(args.valid_path, args.mode, transform=None)
        dataloader = DataLoader(dataset, batch_size=args.batch_size_not_train, shuffle=True, num_workers=args.workers, drop_last=True)

        print("--- create valid dataloader ---")
        print(len(dataset), ", valid dataset created")
        print(len(dataloader), ", valid dataloaders created")

        return dataloader

    else:  # args.mode=='pred'
        dataset = FaultDataset(args.pred_path, args.mode, transform=None)
        dataloader = DataLoader(dataset, batch_size=args.batch_size_not_train, shuffle=False, num_workers=args.workers, drop_last=True)
        print("--- create prediction dataloader ---")
        print(len(dataset), ", prediction dataset created")
        print(len(dataloader), ", prediction dataloaders created")
        return dataloader


def compute_loss(outputs, labels, args):
    if args.loss_func == 'dice':
        criterion = DiceLoss().to(args.device)
        loss = criterion(outputs, labels)

        return loss

    elif args.loss_func == 'cross_with_weight':
        neg = (1 - labels).sum()  # 算有多少个0
        pos = labels.sum()  # 算有多少个1
        beta = neg / (neg + pos)

        weight = torch.tensor([1 - beta, beta]).to(args.device)

        loss = nn.CrossEntropyLoss(weight=weight, reduction='mean')(outputs, labels.long())

        return loss
    else:
        raise ValueError("Only ['DiceLoss', 'CrossEntropyLoss'] loss is supported.")


def con_matrix(outputs, labels, args):
    y_pred = outputs.detach().cpu().numpy()
    y_true = labels.detach().cpu().numpy()

    y_pred = y_pred.argmax(axis=1).flatten()
    y_true = y_true.flatten()

    num_class = args.out_channels
    current = confusion_matrix(y_true, y_pred, labels=range(num_class))  # confusion_matrix混淆矩阵，计算把xxx预测成xxx的次数

    # compute mean iou
    intersection = np.diag(current)
    # 一维数组的形式返回混淆矩阵的对角线元素
    ground_truth_set = current.sum(axis=1)
    # 按行求和
    predicted_set = current.sum(axis=0)
    # 按列求和
    union = ground_truth_set + predicted_set - intersection + 1e-7
    IoU = intersection / union.astype(np.float32)
    union_dice = ground_truth_set + predicted_set + 1e-7
    DICE = 2 * intersection / union_dice.astype(np.float32)

    return np.mean(IoU), np.mean(DICE)


def save_train_info(args, train_RESULT, val_RESULT):
    if not os.path.exists('./EXP/' + args.exp + '/results/train/'):
        os.makedirs('./EXP/' + args.exp + '/results/train/')

    data_df = pd.DataFrame(train_RESULT)
    data_df.columns = ['train_loss', 'train_iou', 'train_dice']
    data_df.index = np.arange(0, args.epochs, 1)
    writer = pd.ExcelWriter('./EXP/' + args.exp + '/results/train/train_result.xlsx')
    data_df.to_excel(writer, 'page_1', float_format='%.5f')
    writer.save()
    writer.close()

    data_df_val = pd.DataFrame(val_RESULT)
    data_df_val.columns = ['val_loss', 'val_iou', 'val_dice']
    data_df_val.index = np.arange(0, args.epochs, 1)
    writer_val = pd.ExcelWriter('./EXP/' + args.exp + '/results/train/val_result.xlsx')
    data_df_val.to_excel(writer_val, 'page_1', float_format='%.5f')
    writer_val.save()


def save_result(args, segs, inputs, gts, val_loss, val_iou, val_dice):
    result_path = './EXP/' + args.exp + '/results/valid/'
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with open(result_path + "valid_final_result.txt", 'a+') as f:
        f.write('valid loss:\t' + str(val_loss) + '\n')
        f.write('valid iou:\t' + str(val_iou) + '\n')
        f.write('valid dice:\t' + str(val_dice) + '\n')

    if not os.path.exists(result_path + '/numpy/'):
        os.makedirs(result_path + '/numpy/')
    if not os.path.exists(result_path + '/picture/'):
        os.makedirs(result_path + '/picture/')

    for i in range(len(inputs)):

        seg = segs[i].argmax(axis=1)
        img = inputs[i]
        gt = gts[i]
        seg = np.squeeze(seg)
        img = np.squeeze(img)
        gt = np.squeeze(gt)
        # save output
        np.save(result_path + '/numpy/' + str(i) + '_seg.npy', seg)
        np.save(result_path + '/numpy/' + str(i) + '_img.npy', img)
        np.save(result_path + '/numpy/' + str(i) + '_gt.npy', gt)
        # save picture

        index = np.arange(0, 128, 50)
        for idx in index:
            # dim 0
            plt.subplot(1, 3, 1)
            plt.imshow(img[idx, :, :])
            plt.axis('off')
            plt.title('Image')

            plt.subplot(1, 3, 2)
            plt.imshow(gt[idx, :, :])
            plt.axis('off')
            plt.title('Ground Truth')

            plt.subplot(1, 3, 3)
            plt.imshow(seg[idx, :, :])
            plt.axis('off')
            plt.title('Segmentation')

            plt.savefig(result_path + '/picture/No_' + str(i) + '_idx_' + str(idx) + '_dim_0.png')
            plt.close()
            # dim 1
            plt.subplot(1, 3, 1)
            plt.imshow(img[:, idx, :])
            plt.axis('off')
            plt.title('Image')

            plt.subplot(1, 3, 2)
            plt.imshow(gt[:, idx, :])
            plt.axis('off')
            plt.title('Ground Truth')

            plt.subplot(1, 3, 3)
            plt.imshow(seg[:, idx, :])
            plt.axis('off')
            plt.title('Segmentation')

            plt.savefig(result_path + '/picture/No_' + str(i) + '_idx_' + str(idx) + '_dim_1.png')
            plt.close()
            # dim 2
            plt.subplot(1, 3, 1)
            plt.imshow(img[:, :, idx])
            plt.axis('off')
            plt.title('Image')

            plt.subplot(1, 3, 2)
            plt.imshow(gt[:, :, idx])
            plt.axis('off')
            plt.title('Ground Truth')

            plt.subplot(1, 3, 3)
            plt.imshow(seg[:, :, idx])
            plt.axis('off')
            plt.title('Segmentation')

            plt.savefig(result_path + '/picture/No_' + str(i) + '_idx_' + str(idx) + '_dim_2.png')
            plt.close()


def load_pred_data(args):
    if args.pred_data_name == 'f3':
        print("Data use f3.")
        data = np.load('f3_data_path')
        return data
    elif args.pred_data_name == 'kerry':
        print("Data use kerry.")
        data = np.load('kerry_data_path')
        return data
    else:
        raise ValueError("Only ['f3', 'kerry'] mode is supported.")


def save_pred_picture(gx, gy, save_path, pred_data_name):
    k1, k2, k3 = 80, 80, 80
    gx1 = gx[k1, :, :]
    gy1 = gy[k1, :, :]
    gx2 = gx[:, k2, :]
    gy2 = gy[:, k2, :]
    gx3 = gx[:, :, k3]
    gy3 = gy[:, :, k3]

    # xline slice
    plt.subplot(1, 2, 1)
    plt.imshow(gx1, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(gy1, cmap='gray')

    plt.savefig(save_path + pred_data_name + '_dim_0.png', dpi=600)

    # inline slice
    plt.subplot(1, 2, 1)
    plt.imshow(gx2, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(gy2, cmap='gray')

    plt.savefig(save_path + pred_data_name + '_dim_1.png', dpi=600)

    # time slice
    plt.subplot(1, 2, 1)
    plt.imshow(gx3, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(gy3, cmap='gray')

    plt.savefig(save_path + pred_data_name + '_dim_2.png', dpi=600)
