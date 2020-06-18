import os

from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

from transforms import RandomCrop, RandomHorizontalFlip, Resize, ToTensor, Normalize, GenOneHotLabel
from constants import imagenet_stats


class BatchDataset(Dataset):
    def __init__(self, img_root, label_root, file_list, transform=None):
        self.transform = transform

        with open(file_list, 'r') as f:
            image_names = f.readlines()
        image_names = [x.strip() for x in image_names]
        image_names = sorted(image_names)
        self.image_names = [os.path.join(img_root, x + '.jpg') for x in image_names]
        self.label_names = [os.path.join(label_root, x + '.png') for x in image_names]

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        label_name = self.label_names[idx]

        image = Image.open(image_name)
        label = Image.open(label_name)

        sample = {
            'image': image,
            'label': label,  # one hot label
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.image_names)

    @staticmethod
    def get_training_loader(img_root, label_root, file_list, batch_size, image_size):
        transformed_dataset = BatchDataset(
            img_root, label_root, file_list,
            transform=transforms.Compose([
                RandomHorizontalFlip(),
                # RandomCrop(image_size),
                # Resize(image_size),
                Normalize(
                    imagenet_stats['mean'],
                    imagenet_stats['std']
                ),
                GenOneHotLabel(),
            ])
        )
        loader = DataLoader(
            transformed_dataset,
            batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=False,
        )
        return loader

    @staticmethod
    def get_testing_loader(img_root, label_root, file_list, batch_size, image_size):
        transformed_dataset = BatchDataset(
            img_root, label_root, file_list,
            transform=transforms.Compose([
                # Resize(image_size),
                ToTensor(),
                Normalize(
                    imagenet_stats['mean'],
                    imagenet_stats['std']
                ),
                GenOneHotLabel(),
            ])
        )
        loader = DataLoader(
            transformed_dataset,
            batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        return loader


