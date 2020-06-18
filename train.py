import argparse


import torch
import scipy.io


from models import unet
from data import BatchDataset

parser = argparse.ArgumentParser()
parser.add_argument('--image-root', default='./VOC2012/JPEGImages', help='path to input images')
parser.add_argument('--label-root', default='./VOC2012/SegmentationClass', help='path to input images')
parser.add_argument('--file-list', default='./VOC2012/ImageSets/Segmentation/val.txt', help='path to input images')
parser.add_argument('--experiment', default='test', help='the path to store sampled images and models')
parser.add_argument('--model-root', default='checkpoint', help='the path to store the testing results')
parser.add_argument('--epochs', type=int, default=210, help='the number of epochs being trained')
parser.add_argument('--batch', type=int, default=1, help='the size of a batch')
parser.add_argument('--class', type=int, default=21, help='the number of classes')
parser.add_argument('--model', default='unet', choices=['unet', 'spp', 'dilation'], help='the model to use')
parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='the device to use')
parser.add_argument('--colormap', default='colormap.mat', help='colormap for visualization')

parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')

parser.add_argument('--checkpoint', default='./checkpoint.pth.tar', help='previous checkpoints')


def main():
    global args
    
    args = parser.parse_args()
    print(args)

    colormap = scipy.io.loadmat(args.colormap)['cmap']

    assert (args.batchSize == 1)

    args.experiment += '_' + args.model
    args.model_root += '_' + args.model

    params = dict()
    params['model'] = unet.imodel(args.model).to(args.device)
    params['optimizer'] = torch.optim.Adam(params['model'].parameters(), args.lr, weight_decay=args.wd)
    params['epoch'] = 0

    train_loader = BatchDataset.get_training_loader(
        args.img_root,
        args.label_root,
        args.file_list,
        args.batch_size,
        0
    )

    while params['epoch'] < args.epochs:
        train(train_loader, params['model'], params['optimizer'], params['epoch'])
        save_model(params, args.checkpoint)
        params['epoch'] += 1


def save_model(params, path):
    states = dict()
    for k in params:
        states[k] = params[k].state_dict()
    torch.save(states, path)


def train(train_loader, model, optimizer, epoch):
    for i, sample_batched in enumerate(train_loader):
        image, label = sample_batched['image'], sample_batched['label']
        image = torch.autograd.Variable(image).to(args.device)
        label = torch.autograd.Variable(label).to(args.device)

        optimizer.zero_grad()
        pred = model(image)

        loss = torch.mean(pred * label)
        loss.backward()
        optimizer.step()
