# import argparse
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


dd = dict()
dd['aa'] = 233

bb = dict(**dd)
bb['bb'] = 244
bb['aa'] = 211
print(bb)

for k in bb:
    print(k)
