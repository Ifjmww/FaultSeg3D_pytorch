import os
from utils.tools import save_pred_picture, load_pred_data
from models.faultseg3d import FaultSeg3D
from matplotlib import pyplot as plt
import numpy as np
import torch


# set gaussian weights in the overlap bounaries
# 在重叠边界中设置高斯权重
def getMask(overlap, n1, n2, n3):
    sc = np.zeros((n1, n2, n3), dtype=np.single)
    sc = sc + 1
    sp = np.zeros((overlap), dtype=np.single)
    sig = overlap / 4
    sig = 0.5 / (sig * sig)
    for ks in range(overlap):
        ds = ks - overlap + 1
        sp[ks] = np.exp(-ds * ds * sig)
    for k1 in range(overlap):
        for k2 in range(n2):
            for k3 in range(n3):
                sc[k1][k2][k3] = sp[k1]
                sc[n1 - k1 - 1][k2][k3] = sp[k1]
    for k1 in range(n1):
        for k2 in range(overlap):
            for k3 in range(n3):
                sc[k1][k2][k3] = sp[k2]
                sc[k1][n3 - k2 - 1][k3] = sp[k2]
    for k1 in range(n1):
        for k2 in range(n2):
            for k3 in range(overlap):
                sc[k1][k2][k3] = sp[k3]
                sc[k1][k2][n3 - k3 - 1] = sp[k3]
    # return np.transpose(sc)
    return sc


def pred(args):
    print("======================= test ==============================")
    # load and create model
    model = FaultSeg3D(args.in_channels, args.out_channels).to(args.device)

    model_path = './EXP/' + args.exp + '/models/' + args.pretrained_model_name

    model.load_state_dict(torch.load(model_path))
    print("Loaded model from disk")

    # training image dimensions
    n1, n2, n3 = args.window_size

    # 加载数据
    gx = load_pred_data(args)
    m1, m2, m3 = gx.shape[0], gx.shape[1], gx.shape[2]

    args.overlap = 12  # overlap width
    c1 = np.round((m1 + args.overlap) / (n1 - args.overlap) + 0.5)
    c2 = np.round((m2 + args.overlap) / (n2 - args.overlap) + 0.5)
    c3 = np.round((m3 + args.overlap) / (n3 - args.overlap) + 0.5)

    c1 = int(c1)
    c2 = int(c2)
    c3 = int(c3)

    p1 = (n1 - args.overlap) * c1 + args.overlap
    p2 = (n2 - args.overlap) * c2 + args.overlap
    p3 = (n3 - args.overlap) * c3 + args.overlap

    gp = np.zeros((p1, p2, p3), dtype=np.single)
    gy = np.zeros((p1, p2, p3), dtype=np.single)
    mk = np.zeros((p1, p2, p3), dtype=np.single)
    gs = np.zeros((1, 1, n1, n2, n3), dtype=np.single)
    gp[0:m1, 0:m2, 0:m3] = gx
    sc = getMask(args.overlap, n1, n2, n3)

    print('>>>Start Predicting<<<')
    count = 0
    for k3 in range(c3):
        for k2 in range(c2):
            for k1 in range(c1):
                count += 1
                print('[', count, ' / ', (c1 * c2 * c3), '] ====================================')

                b1 = k1 * n1 - k1 * args.overlap
                e1 = b1 + n1
                b2 = k2 * n2 - k2 * args.overlap
                e2 = b2 + n2
                b3 = k3 * n3 - k3 * args.overlap
                e3 = b3 + n3
                gs[0, 0, :, :, :] = gp[b1:e1, b2:e2, b3:e3]
                gs_m = np.mean(gs)
                gs_s = np.std(gs)
                #
                gs = (gs - gs_m) / gs_s
                inputs = torch.from_numpy(gs).to(args.device)
                y = model(inputs)
                outputs = y.argmax(axis=1)
                outputs = outputs.detach().cpu().numpy()
                outputs = np.squeeze(outputs)

                gy[b1:e1, b2:e2, b3:e3] = gy[b1:e1, b2:e2, b3:e3] + outputs[:, :, :] * sc
                mk[b1:e1, b2:e2, b3:e3] = mk[b1:e1, b2:e2, b3:e3] + sc
    gy = gy / mk
    gy = gy[0:m1, 0:m2, 0:m3]

    print("---Start Save results  ······")
    save_path = './EXP/' + args.exp + '/results/pred/' + args.pred_data_name + '/'
    if not os.path.exists(save_path + '/numpy/'):
        os.makedirs(save_path + '/numpy/')
    if not os.path.exists(save_path + '/picture/'):
        os.makedirs(save_path + '/picture/')
    np.save(save_path + '/numpy/' + args.pred_data_name + '.npy', gy)

    save_pred_picture(gx, gy, save_path + '/picture/', args.pred_data_name)
