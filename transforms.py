import random
import torch
import torch.nn.functional as F

from PIL import Image
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
        image = transforms.Normalize(self.mean, self.std)(image)

        sample['image'] = image
        return sample


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        i, j, h, w = transforms.RandomCrop.get_params(image, self.size)
        image = image.crop(j, i, j + w, i + h)
        label = label.crop(j, i, j + w, i + h)

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
        label = transforms.ToTensor()(label)

        sample['image'] = image
        sample['label'] = label
        return sample


class GenOneHotLabel(object):
    def __call__(self, sample):
        label = sample['label'].type(torch.LongTensor)
        label = label.squeeze(0)
        label = F.one_hot(label, num_classes=21).permute(2, 0, 1)
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
        GenOneHotLabel(),
        # AddRequireGrad(),
    ])

    sample = trans(sample)
    print(sample['label'].size())

    image = torch.autograd.Variable(sample['label']).to('cpu')
    print(image)

