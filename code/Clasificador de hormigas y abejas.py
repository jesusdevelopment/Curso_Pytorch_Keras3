#%% [markdown]
# # Clasificador de hormigas y abejas
# Bibliotecas
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms, utils
from torchvision.models import ResNet50_Weights
import numpy as np
import matplotlib.pyplot as plt
import time
import copy
import os
#%%[markdown]
# DataSet
# Download the dataset
if not os.path.exists('hymenoptera_data'):
    !wget -q https://download.pytorch.org/tutorial/hymenoptera_data.zip
    !unzip -q -o hymenoptera_data.zip
    !rm hymenoptera_data.zip
#%% [markdown]
# Configuración de dispositivos
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Transformaciones para los datos
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
#%%
!ls
# %% [markdown]
# # Carga de datos
data_set='hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_set, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=4, shuffle=True,  num_workers=4, pin_memory=True),
    'val':   torch.utils.data.DataLoader(image_datasets['val'],   batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
# %%[markdown]
# Visualización de algunas imágenes

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))
# Make a grid from batch
out = utils.make_grid(inputs)
imshow(out, title=[class_names[x] for x in classes])
# %%
