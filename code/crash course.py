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

# %% [markdown]
# ## 4. Carga de Datos
train_DataSet=TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_DataSet, batch_size=32, shuffle=True)

# %% [markdown]
# ## 5. Creación del Modelo

class SimpleModel(nn.Module):
    def __init__(self, input_dim):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x


model = SimpleModel(input_dim=X_train_tensor.shape[1])
# %% [markdown]
# ## 6. Definición de la Función de Pérdida y Optimizador
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# %%[markdown]
# ## 7. Entrenamiento del Modelo
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.unsqueeze(1))