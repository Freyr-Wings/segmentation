import argparse
#
# parser = argparse.ArgumentParser()
# # The locationi of training set
# parser.add_argument('--imageRoot', default='./VOC2012/JPEGImages', help='path to input images')
# parser.add_argument('--labelRoot', default='./VOC2012/SegmentationClass', help='path to input images')
# parser.add_argument('--fileList', default='./VOC2012/ImageSets/Segmentation/val.txt', help='path to input images')
# parser.add_argument('--experiment', default='test', help='the path to store sampled images and models')
# parser.add_argument('--modelRoot', default='checkpoint', help='the path to store the testing results')
# parser.add_argument('--epochId', type=int, default=210, help='the number of epochs being trained')
# parser.add_argument('--batchSize', type=int, default=1, help='the size of a batch')
# parser.add_argument('--numClasses', type=int, default=21, help='the number of classes')
# parser.add_argument('--isDilation', action='store_true', help='whether to use dialated model or not')
# parser.add_argument('--isSpp', action='store_true', help='whether to do spatial pyramid or not')
# parser.add_argument('--noCuda', action='store_true', help='do not use cuda for training')
# parser.add_argument('--gpuId', type=int, default=0, help='gpu id used for training the network')
# parser.add_argument('--colormap', default='colormap.mat', help='colormap for visualization')


# parser = argparse.ArgumentParser()
#
# parser.add_argument('--experiment', default='test', help='the path to store sampled images and models')
# parser.add_argument('--model-root', default='./checkpoint.pth.tar', help='the path to store the testing results')
#
# parser.add_argument('--colormap', default='colormap.mat', help='colormap for visualization')
#
#
# parser.add_argument('--checkpoint', default='./checkpoint.pth.tar', help='previous checkpoints')
#
#
# # done
# parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'], help='the device to use')
#
# parser.add_argument('--image-root', default='./VOC2012/JPEGImages', help='path to input images')
# parser.add_argument('--label-root', default='./VOC2012/SegmentationClass', help='path to input images')
# parser.add_argument('--file-list', default='./VOC2012/ImageSets/Segmentation/val.txt', help='path to input images')
#
# parser.add_argument('--model-name', default='unet', choices=['unet', 'spp', 'dilation'], help='the model to use')
#
# parser.add_argument('--num-class', type=int, default=21, help='the number of classes')
# parser.add_argument('--epochs', type=int, default=210, help='the number of epochs being trained')
# parser.add_argument('--batch-size', type=int, default=1, help='the size of a batch')
# parser.add_argument('--lr', default=0.0005, type=float, help='initial learning rate')
# parser.add_argument('--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')
#
# args = parser.parse_args()
# print(args)
#
# if args.model_root is not None:
#     print("???")
#
# import os
# print(os.path.exists("./loss.py"))


# dd = dict()
# dd['aa'] = 233
#
# bb = dict(**dd)
# bb['bb'] = 244
# bb['aa'] = 211
# print(bb)
#
# for k in bb:
#     print(k)

import numpy as np
from sklearn.metrics import confusion_matrix
import torch
#
label = torch.randint(0, 10, (1, 5, 5)).long()
print(label)
output = torch.randn(1, 5, 5)
# predict = torch.argmax(output, dim=1, keepdim=True)
# print(predict.size())
label_np = label.numpy()
output_np = output.numpy()
# print(label_np.shape)
# print(output_np.shape)
#
# from train import IoU
#
# iou = IoU(10)
#
# iou.add_batch(label_np, output_np)
# print(iou.get_mean_iou())


# import scipy.io
# colormap = scipy.io.loadmat("./datas/colormap.mat")['cmap']
# print(colormap)
# size = label_np.shape
# color_image = np.zeros((size[1], size[2], 3))
# print(colormap[label_np[0]])
# res = np.clip(colormap[label_np[0]] * 255, a_min=0, a_max=255)
# print(res)
#
# from matplotlib import pyplot as plt
#
# plt.imshow(res)
# plt.show()

# from train import Epoch
#
# e = Epoch()
# print(e)
# e.step()
# print(int(e)*2)
import os
print(os.path.basename('../Users/ck'))


import fastai
from fastai import *
from fastai.vision import *

learn = unet_learner(data,models.resnet34,metrics=custom_acc,loss_func=custom_loss)

SegmentationItemList
