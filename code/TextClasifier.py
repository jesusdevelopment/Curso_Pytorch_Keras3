# %% [mark down]
# 1. Importación de Bibliotecas para Clasificación de Texto
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# %% [markdown]
# 2. Definición del Modelo de Clasificación

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
# %% [markdown]
# 3. Definición del método forward 

    def forward(self, text):
            embedded = self.embedding(text)
            # Pasar por la capa LSTM
            output, (hidden, cell) = self.lstm(embedded)
            # Usamos la última capa oculta
            final_hidden = hidden[-1]
            return self.fc(final_hidden)
    
# %% [markdown]
# 4. Instanciación del modelo
        
vocab_size=10000
embed_dim = 100
hidden_dim = 256
output_dim = 2

# %% [markdown]
# 5. Entrenamiento del modelo
model=TextClassifier(vocab_size, embed_dim, hidden_dim, output_dim)

model
