import os
import argparse

import torch
import torch.nn as nn
import scipy.io
import numpy as np

from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam, lr_scheduler
from models import unet
from data import VOCDataset


# python3 train.py --colormap=./datas/colormap.mat --image-root=/content/gdrive/My\ Drive/datasets/PascalVOC/VOCdevkit/VOC2012/JPEGImages --label-root=/content/gdrive/My\ Drive/datasets/PascalVOC/VOCdevkit/VOC2012/SegmentationClass --train-list=/content/gdrive/My\ Drive/datasets/PascalVOC/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt --valid-list=/content/gdrive/My\ Drive/datasets/PascalVOC/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt
parser = argparse.ArgumentParser()

parser.add_argument('--experiment', default='test', help='the path to store sampled images and models')
parser.add_argument('--model-root', default='./checkpoint.pth.tar', help='the path to store the testing results')

parser.add_argument('--colormap', default='./datas/colormap.mat', help='colormap for visualization')

# done
parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='the device to use')

parser.add_argument('--image-root', default='./VOC2012/JPEGImages', help='path to input images')
parser.add_argument('--label-root', default='./VOC2012/SegmentationClass', help='path to input images')
parser.add_argument('--train-list', default='./VOC2012/ImageSets/Segmentation/train.txt', help='path to train images')
parser.add_argument('--valid-list', default='./VOC2012/ImageSets/Segmentation/val.txt', help='path to validation images')

parser.add_argument('--model-name', default='unet', choices=['unet', 'spp', 'dilation'], help='the model to use')

parser.add_argument('--num-class', type=int, default=21, help='the number of classes')
parser.add_argument('--epochs', type=int, default=210, help='the number of epochs being trained')
parser.add_argument('--batch-size', type=int, default=1, help='the size of a batch')
parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')

parser.add_argument(
    '--checkpoint',
    default='./checkpoint',
    help='path to previous checkpoints, if not exist then we will automatically create one'
)
parser.add_argument('--tb-root', default='./tblogdir', help='path to tensorboard log directory')


def main():
    global args
    
    args = parser.parse_args()
    print(args)

    global cmap
    cmap = scipy.io.loadmat(args.colormap)['cmap']
    cmap = torch.from_numpy(cmap)

    args.checkpoint += args.model_name + '_' + args.checkpoint

    global tb_writer
    tb_writer = SummaryWriter(args.tb_root)

    params = dict()
    params['model'] = unet.imodel(args.model_name).to(args.device)
    params['optimizer'] = Adam(
        params['model'].parameters(),
        args.lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.wd
    )
    params['epoch'] = Epoch()
    params['scheduler'] = lr_scheduler.StepLR(params['optimizer'], step_size=100, gamma=0.92)

    if os.path.exists(args.checkpoint):
        load_model(params, args.checkpoint)

    train_loader = VOCDataset.get_training_loader(
        args.image_root,
        args.label_root,
        args.train_list,
        args.batch_size,
        512,
        512,
        args.num_class,
    )

    valid_loader = VOCDataset.get_testing_loader(
        args.image_root,
        args.label_root,
        args.valid_list,
        args.batch_size,
        512,
        args.num_class,
    )

    criterion = nn.CrossEntropyLoss()

    while params['epoch'] < args.epochs:
        train(train_loader, params['model'], params['optimizer'], criterion, params['epoch'])
        validate(valid_loader, params['model'], criterion, params['epoch'])

        params['epoch'].step()
        params['scheduler'].step()

        save_model(params, args.checkpoint)


def save_model(params, path):
    states = dict()
    for k in params:
        states[k] = params[k].state_dict()
    torch.save(states, path)


def load_model(params, path):
    states = torch.load(path)
    for k in params:
        params[k].load_state_dict(states[k])


def train(loader, model, optimizer, criterion, epoch):
    print("----- TRAINING - EPOCH", epoch, "-----")
    model.train()
    num_batch = len(loader)
    iou = IoU(args.num_class)
    total_loss = 0.

    for i, sample_batched in enumerate(loader):
        image, label = sample_batched['image'], sample_batched['label']
        image = torch.autograd.Variable(image).to(args.device)
        label = torch.autograd.Variable(label).to(args.device)

        optimizer.zero_grad()
        output = model(image)

        loss = criterion(output, label)

        loss.backward()
        optimizer.step()

        predict = torch.argmax(output, dim=1, keepdim=True)
        label_np = label.cpu().numpy().flatten()
        output_np = predict.cpu().numpy().flatten()

        iou.add_batch(label_np, output_np)
        total_loss += loss.data[0]

        tb_writer.add_scalar('Loss/train', loss.data[0], epoch * num_batch + i)
        tb_writer.add_scalar('Mean IoU/train', iou.get_mean_iou(), epoch * num_batch + i)
        print('Epoch %d iteration %d: Loss %.5f Accumulated Loss %.5f, mIoU %.5f'.format(
            epoch, i, loss.data[0], total_loss/(i+1), iou.get_mean_iou()
        ))


def validate(loader, model, criterion, epoch):
    print("----- VALIDATING - EPOCH", epoch, "-----")

    model.eval()
    iou = IoU(args.num_class)
    total_loss = 0.

    for i, sample_batched in enumerate(loader):
        image, label = sample_batched['image'], sample_batched['label']
        image = torch.autograd.Variable(image).to(args.device)
        label = torch.autograd.Variable(label).to(args.device)

        output = model(image)
        loss = criterion(output, label)

        predict = torch.argmax(output, dim=1, keepdim=True)
        label_np = label.cpu().numpy().flatten()
        output_np = predict.cpu().numpy().flatten()

        iou.add_batch(label_np, output_np)
        total_loss += loss.data[0]

        if i == 0:
            predicts = predict.cpu().long()
            for t in range(image.size(0)):
                predict = predicts[t]
                colored_img = cmap[predict[0]]
                tb_writer.add_image('images/%d'.format(t), colored_img, epoch, dataformats='HWC')

    tb_writer.add_scalar('Loss/val', total_loss, epoch)
    tb_writer.add_scalar('Mean IoU/val', iou.get_mean_iou(), epoch)
    print('Epoch %d: Total Loss %.5f, mIoU %.5f'.format(epoch, total_loss, iou.get_mean_iou()))


class Epoch(int):
    def __init__(self):
        super().__init__()
        self.epoch = 0

    def __int__(self):
        return self.epoch

    def state_dict(self):
        return {"epoch": self.epoch}

    def load_state_dict(self, state_dict: dict):
        self.epoch = state_dict["epoch"]

    def step(self):
        self.epoch += 1


class IoU:
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_mat = np.zeros((num_class, num_class), dtype=np.int64)

    def get_mean_iou(self):
        intersection = np.diag(self.confusion_mat)
        ground_truth_set = self.confusion_mat.sum(axis=1)
        predicted_set = self.confusion_mat.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection

        intersection_over_union = intersection / union.astype(np.float32)
        return np.mean(intersection_over_union)

    def add_batch(self, label_np, output_np):
        assert(label_np.max() < self.num_class)
        assert(output_np.max() < self.num_class)
        cm = confusion_matrix(label_np, output_np, list(range(self.num_class)))
        self.confusion_mat += cm


if __name__ == '__main__':
    main()
