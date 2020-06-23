import random
import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image, ImageOps
from torchvision import transforms

from constants import imagenet_stats


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = transforms.Resize(self.size, Image.BILINEAR)(image)
        label = transforms.Resize(self.size, Image.NEAREST)(label)

        sample['image'] = image
        sample['label'] = label
        return sample


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        print(image)
        image = transforms.Normalize(self.mean, self.std)(image)

        sample['image'] = image
        return sample


class RandomTranslation(object):
    def __init__(self, translation=2):
        self.translation = translation

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        trans_x = random.randint(-self.translation, self.translation)
        trans_y = random.randint(-self.translation, self.translation)
        image = ImageOps.expand(image, border=(trans_x, trans_y, 0, 0), fill=0)
        label = ImageOps.expand(label, border=(trans_x, trans_y, 0, 0), fill=255)  # pad label filling with 255
        image = image.crop((0, 0, image.size[0] - trans_x, image.size[1] - trans_y))
        label = label.crop((0, 0, label.size[0] - trans_x, label.size[1] - trans_y))

        sample['image'] = image
        sample['label'] = label
        return sample


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        i, j, h, w = transforms.RandomCrop.get_params(image, self.size)
        image = image.crop((j, i, j + w, i + h))
        label = label.crop((j, i, j + w, i + h))

        sample['image'] = image
        sample['label'] = label
        return sample


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if random.random() < self.p:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        sample['image'] = image
        sample['label'] = label
        return sample


class ToTensor(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = transforms.ToTensor()(image)
        # nplabel = np.array(label)
        # print(nplabel)
        # print("npmin", np.unique(nplabel))
        # nplabel[nplabel == 15] = 2
        # nplabel[nplabel == 255] = 0
        # import matplotlib.pyplot as plt
        # plt.imshow(nplabel)
        # plt.colorbar()
        # plt.show()
        label = torch.from_numpy(np.array(label)).long()
        label[label == 255] = 0  # set contour as background
        sample['image'] = image
        sample['label'] = label
        return sample


class GenOneHotLabel(object):
    def __init__(self, num_class):
        self.num_class = num_class

    def __call__(self, sample):
        label = sample['label']
        label = F.one_hot(label, num_classes=self.num_class).permute(2, 0, 1)
        sample['label'] = label
        return sample


class AddRequireGrad(object):
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image.requires_grad = True
        label.requires_grad = True
        sample['image'] = image
        sample['label'] = label
        return sample


if __name__ == '__main__':
    image_name = 'datas/images/2007_000032.jpg'
    label_name = 'datas/masks/2007_000032.png'

    image = Image.open(image_name)
    label = Image.open(label_name)

    sample = {
        'image': image,
        'label': label,  # one hot label
    }

    trans = transforms.Compose([
        RandomHorizontalFlip(),
        # Resize(image_size),
        ToTensor(),
        Normalize(
            imagenet_stats['mean'],
            imagenet_stats['std']
        ),
        GenOneHotLabel(21),
        # AddRequireGrad(),
    ])

    sample = trans(sample)
    print(sample['label'].size())

    processed_label = sample['label']
    print("class 0:", torch.sum(processed_label[0]))
    print("class 1:", torch.sum(processed_label[1]))
    print("class 2:", torch.sum(processed_label[2]))
    print("class 21:", torch.sum(processed_label[21]))

    image = torch.autograd.Variable(sample['label']).to('cpu')
    print(image)

    autograd_tensor = torch.randn((2, 3, 4), requires_grad=True)
    print(autograd_tensor)


