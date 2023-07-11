import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from glob import glob
from skimage import io
import re
import piexif
from PIL import Image, ExifTags
from pillow_heif import register_heif_opener
from datetime import datetime
from PIL import Image
import pillow_heif
class HarryOllieDataset(Dataset):

    def __init__(self, olliefolder, harryfolder, transform=None):
        self.ollie = olliefolder
        self.harry = harryfolder
        self.transform = transform
        self.files, self.names = self.preprocessing(olliefolder, "ollie")
        harry_images, harry_names = self.preprocessing(harryfolder, "harry")
        self.files += harry_images
        self.names += harry_names


    def preprocessing(self, folder, name):
        files = glob(os.path.join(folder, "*"))
        names = [name] * len(files)
        return files, names

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = self.files[idx]
        if "HEIC" in item:
            pillow_heif.register_heif_opener()
            image = Image.open(item)
            image = np.asarray(image) / 255.0

        else:
            image = io.imread(item) / 255.0

        image = image[:, :, :3]

        if self.transform is not None:
            image = self.transform(image)

        if self.names[idx] == "ollie":
            label = 0
        else:
            label = 1

        return image, torch.tensor(label)

if __name__ == "__main__":
    dataset = HarryOllieDataset("ollieimages", "harryimages")
    print(dataset[0])

