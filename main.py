import TransferLearningTemplate as tlt
from ollieharrydataset import HarryOllieDataset
from torchvision import transforms
import torch
from skimage import io
import matplotlib.pyplot as plt
from pillow_heif import register_heif_opener
from datetime import datetime
from PIL import Image
import pillow_heif
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import os

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                #transforms.RandomResizedCrop((224, 224)),
                transforms.RandomHorizontalFlip(),


            ])

dataset = HarryOllieDataset('ollieimages', 'harryimages', transform)
# tlt.Visualize_Dataset(dataset)

train_dataloader, test_dataloader = tlt.Create_DataLoader(dataset, batch_size=8)

net, loss_fn = tlt.MakeModel(num_classes=2, loss_type = "notdistance")
net = net.float()
checkpoint = torch.load("checkpoint.pt")
net.load_state_dict(checkpoint)

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
inp = askopenfilename( initialdir = os.getcwd()) # show an "Open" dialog box and return the path to the selected file

try:
    image = io.imread(inp) / 255.0
except:
    pillow_heif.register_heif_opener()
    image = Image.open(inp)
    image = np.asarray(image) / 255.0

image = image[:, :, :3]
image = transform(image)
plt.imshow(image.permute(1, 2, 0))
plt.show()
image = image.unsqueeze(0)
output = net(image.float())
print(output)


# tlt.train(net, loss_fn, train_dataloader, test_dataloader, epochs=150 )
