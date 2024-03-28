import os
import torch
from tqdm import tqdm
from utils.tools import load_data, compute_loss, con_matrix, save_train_info, save_result
import torch.optim as optim
from models.faultseg3d import FaultSeg3D
import numpy as np


def train(args):
    # set device
    device = torch.device(args.device)
    print("---")
    print('Device is :', device)
    # Load data
    print("---")
    print("Loading data ... ")
    train_loader, val_loader = load_data(args)
    print('Create model...')
    model = FaultSeg3D(args.in_channels, args.out_channels).to(args.device)
    # Initialize optimizer
    print("---")
    print("Define optimizer ... ")

    optimizer = optim.Adam(model.parameters(), lr=args.optim_lr)

    # Set model save path   ./EXP/<exp>/models/
    model_path = './EXP/' + args.exp + '/models/'
    print("---")
    print("The model is saved in : ", model_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # start training
    print("---")
    print("Start training ... ")

    train_RESULT = []
    val_RESULT = []

    best_iou = 0.0

    for epoch in range(args.epochs):

        model.train()
        # 训练模式
        train_loss = 0.0
        train_iou = 0.0
        train_dice = 0.0

        for step, data in enumerate(tqdm(train_loader, desc='[Train] Epoch' + str(epoch + 1) + '/' + str(args.epochs))):
            inputs, labels = data['x'].to('cuda'), data['y'].to('cuda')

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = compute_loss(outputs, labels, args)
            iou, dice = con_matrix(outputs, labels, args)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou += iou
            train_dice += dice

        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for step, data in enumerate(tqdm(val_loader, desc='[VALID] Valid ')):
                inputs = data['x'].to('cuda')
                labels = data['y'].to('cuda')
                outputs = model(inputs)
                loss = compute_loss(outputs, labels, args)
                iou, dice = con_matrix(outputs, labels, args)

                val_loss += loss.item()
                val_iou += iou
                val_dice += dice
        print(
            " train loss: {:.4f}".format(train_loss / len(train_loader)),
            " train iou: {:.4f}".format(train_iou / len(train_loader)),
            " train dice:{:.4f}".format(train_dice / len(train_loader)),
            " val loss: {:.4f}".format(val_loss / len(val_loader)),
            " val iou: {:.4f}".format(val_iou / len(val_loader)),
            " val dice:{:.4f}".format(val_dice / len(val_loader))
        )

        train_result = np.append(train_loss / len(train_loader), [train_iou / len(train_loader), train_dice / len(train_loader)])
        train_RESULT.append(train_result)

        val_result = np.append(val_loss / len(val_loader), [val_iou / len(val_loader), val_dice / len(val_loader)])
        val_RESULT.append(val_result)

        if (val_iou / len(val_loader)) > best_iou:
            print("new best ({:.6f} --> {:.6f}). ".format(best_iou, val_iou / len(val_loader)))
            best_iou = val_iou / len(val_loader)
            best_model_name = 'FaultSeg3D_BEST.pth'.format(epoch + 1, val_iou / len(val_loader))
            torch.save(model.state_dict(), model_path + best_model_name)

        if (epoch + 1) % args.val_every == 0:
            model_name = 'FaultSeg3D_epoch_{}_iou_{:.4f}_CP.pth'.format(epoch + 1, val_iou / len(val_loader))  # CP means checkpoints
            torch.save(model.state_dict(), model_path + model_name)

    # Save training information

    print("---")
    print("Save training information ... ")
    save_train_info(args, train_RESULT, val_RESULT)
    print("---")
    print("Train Finish ! ")
    print("---")
    print("---")
    print("Last validation ... ")
    valid(args, val_loader)

    return 0


def valid(args, val_loader=None):

    device = torch.device(args.device)
    print("---")
    print('Device is :', device)
    # Load data
    print("---")
    print("Loading data ... ")
    if args.mode == 'valid_only':
        val_loader = load_data(args)
    # Load Model
    print("---")
    print("Loading Model ... ")
    model = FaultSeg3D(args.in_channels, args.out_channels).to(args.device)

    model_path = './EXP/' + args.exp + '/models/' + args.pretrained_model_name

    model.load_state_dict(torch.load(model_path))

    segs = []
    inputs = []
    gts = []

    print("---")
    print("Start validation ... ")

    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0

    model.eval()
    with torch.no_grad():
        for step, data in enumerate(tqdm(val_loader, desc='[Valid] Valid')):
            x = data['x'].to(args.device)
            y = data['y'].to(args.device)

            outputs = model(x)
            loss = compute_loss(outputs, y, args)
            iou, dice = con_matrix(outputs, y, args)

            val_loss += loss.item()
            val_iou += iou
            val_dice += dice

            segs.append(outputs.detach().cpu().numpy())
            inputs.append(x.detach().cpu().numpy())
            gts.append(y.detach().cpu().numpy())

        print(
            " val loss: {:.4f}".format(val_loss / len(val_loader)),
            " val iou: {:.4f}".format(val_iou / len(val_loader)),
            " val dice:{:.4f}".format(val_dice / len(val_loader)),
        )

        print("---")
        print("Save result of validation ... ")

        save_result(args, segs, inputs, gts, val_loss / len(val_loader), val_iou / len(val_loader), val_dice / len(val_loader))

        print("---")
        print("Save Finished ! ")
