import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


def Visualize_Dataset(dataset):
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        image, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(image.permute(1, 2, 0))
    plt.show()


def Create_DataLoader(Dataset, batch_size=8, test_ratio=0.25, seed=42):
    N = len(Dataset)
    train_size = int(N * (1 - test_ratio))
    train_set, test_set = torch.utils.data.random_split(Dataset, [train_size, N - train_size])

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
    test_datalodaer = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_datalodaer
