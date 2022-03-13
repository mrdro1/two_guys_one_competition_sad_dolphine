import os
import logging

import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import Compose, Lambda, Normalize, AutoAugment, AutoAugmentPolicy

logging.getLogger().setLevel(logging.INFO)


class HappyWhaleDataset(Dataset):
    def __init__(
            self,
            df: pd.DataFrame,
            image_dir: str,
            return_labels=True,
    ):
        self.df = df
        self.images = self.df["image"]
        self.image_dir = image_dir
        self.std = [0.229, 0.224, 0.225]
        self.mean = [0.485, 0.456, 0.406]
        self.image_transform = Compose(
            [
                AutoAugment(AutoAugmentPolicy.IMAGENET),
                # WARNING hard coded normalization
                Lambda(lambda x: x / 255),
                Normalize(mean=self.mean, std=self.std),

            ]
        )
        logging.warning(f'Used hard coded normalization: mean={self.mean}, std={self.std}')
        self.return_labels = return_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = os.path.join(self.image_dir, self.images.iloc[idx])
        image = read_image(path=image_path)
        image = self.image_transform(image)

        if self.return_labels:
            label = self.df['label'].iloc[idx]
            return image, label
        else:
            return image
