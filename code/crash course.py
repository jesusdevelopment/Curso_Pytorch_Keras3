# %% [markdown]
# ## 1. Importación de Bibliotecas
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %% [markdown]
# ## 2. Descarga y Preparación de los Datos

X,y=load_breast_cancer(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)
print(X_train_scaled.shape)
print(y_train)
print(y_train.shape)
# %% [markdown]
# ## 3. Conversión a Tensores de PyTorch

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
# %% [merkdown]
# ## 4. Carga de Datos y Creación del Modelo

train_DataSet=TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_DataSet, batch_size=32, shuffle=True)



# %%
