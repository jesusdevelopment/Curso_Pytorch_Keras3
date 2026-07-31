# %% [markdown]
# ## 1. Instalación e Importación de Bibliotecas
!pip install --upgrade --force-reinstall fsspec datasets huggingface_hub
 
# %% [markdown]
# ## 2. Importaciones y Configuración de Dispositivo
import time
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Usando el dispositivo: {device}')

# %% [markdown]
# ## 3. Carga y Exploración del Dataset
dataset = load_dataset("fancyzhx/dbpedia_14")

# ## Exploración del Dataset
# 1. Ver qué splits/sets existen
print("Splits disponibles:", dataset.keys())

# 2. Separamos train en Entrenamiento (90%) y Validación (10%)
train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_data = train_val_split["train"]
valid_data = train_val_split["test"]
test_data = dataset["test"]

print(f"Train: {len(train_data):,} ejemplos")
print(f"Valid: {len(valid_data):,} ejemplos")
print(f"Test:  {len(test_data):,} ejemplos")

# 3. Imprimir la estructura global del objeto
print(dataset)

# 4. Ver los nombres de los atributos/columnas
print("Columnas:", dataset["train"].column_names)

# 5. Ver la estructura detallada y tipos de datos (features)
print("Features:", dataset["train"].features)

# 6. Obtener el nombre textual de cada clase/etiqueta (0 a 13)
etiquetas = dataset["train"].features["label"].names
print("\nCategorías de clasificación (0 al 13):")
for idx, nombre in enumerate(etiquetas):
    print(f"  {idx}: {nombre}")

# %% [markdown]
# 8. Primer ejemplo
primer_ejemplo = dataset["train"][0]
print("\nAtributos del primer ejemplo:", list(primer_ejemplo.keys()))
print(f"Etiqueta (Label): {primer_ejemplo['label']}")
print(f"Título: {primer_ejemplo['title']}")
print(f"Texto: {primer_ejemplo['content'][:120]}...")
print("Ejemplo completo:")
print(primer_ejemplo)


# 9. Segundo ejemplo
segundo_ejemplo = dataset["train"][48954]
print("\nAtributos del segundo ejemplo:", list(segundo_ejemplo.keys()))
print(f"Etiqueta (Label): {segundo_ejemplo['label']}")
print(f"Título: {segundo_ejemplo['title']}")
print(f"Texto: {segundo_ejemplo['content'][:120]}...")
print("Ejemplo completo:")
print(segundo_ejemplo)


# %% [markdown]
# ## 4. Tokenizador y Construcción del Vocabulario
def get_tokenizer():
    regex = re.compile(r"\w+|[^\w\s]")
    return lambda texto: regex.findall(texto.lower())

tokenizador = get_tokenizer()

class Vocab:
    def __init__(self, counter, specials=["<unk>"]):
        self.stoi = {}
        self.itos = []
        self.default_index = None
        
        for spec in specials:
            self._add_token(spec)
            
        for token, _ in counter.most_common():
            if token not in self.stoi:
                self._add_token(token)

    def _add_token(self, token):
        self.stoi[token] = len(self.itos)
        self.itos.append(token)

    def set_default_index(self, index):
        self.default_index = index

    def __getitem__(self, token):
        return self.stoi.get(token, self.default_index)

    def __len__(self):
        return len(self.itos)

print("\nProcesando tokens y construyendo vocabulario...")
counter = Counter()
for example in train_data:
    texto = f"{example['title']} {example['content']}"
    counter.update(tokenizador(texto))

vocab = Vocab(counter, specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])
print(f"¡Vocabulario creado! Tamaño total: {len(vocab):,} tokens.")

texto_pipeline = lambda x: [vocab[token] for token in tokenizador(x)]

# %% [markdown]
# ## 5. Función de Agrupamiento (collate_fn)
def collate_batch(batch):
    label_list, text_list = [], []
    offsets = [0]
    
    for item in batch:
        label_list.append(item['label'])
        texto_completo = f"{item['title']} {item['content']}"
        processed_text = torch.tensor(texto_pipeline(texto_completo), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
        
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    
    return label_list.to(device), text_list.to(device), offsets.to(device)

# %% [markdown]
# ## 6. Arquitectura del Modelo e Instanciación
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
    
    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        embedded_norm = self.bn1(embedded)
        embedded_activated = F.relu(embedded_norm)
        return self.fc(embedded_activated)

num_class = len(etiquetas)
embedding_size = 100
vocab_size = len(vocab)

modelo = TextClassificationModel(
    vocab_size=vocab_size, 
    embed_dim=embedding_size, 
    num_class=num_class
).to(device)

print(f"\n¡Modelo instanciado para {num_class} clases!")

# %% [markdown]
# ## 7. Pérdida, Optimizador y DataLoaders
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(modelo.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

# Se crean los 3 dataloaders requeridos
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(valid_data, batch_size=128, shuffle=False, collate_fn=collate_batch)
test_dataloader  = DataLoader(test_data, batch_size=128, shuffle=False, collate_fn=collate_batch)

# %% [markdown]
# ## 8. Funciones de Entrenamiento y Evaluación
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_acc, total_count = 0, 0
    total_loss = 0.0

    for labels, texts, offsets in dataloader:
        optimizer.zero_grad()
        
        predicted_labels = model(texts, offsets)
        loss = criterion(predicted_labels, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_acc += (predicted_labels.argmax(1) == labels).sum().item()
        total_count += labels.size(0)

    return total_loss / total_count, total_acc / total_count

def evaluate(model, dataloader, criterion):
    model.eval()
    total_acc, total_count = 0, 0
    total_loss = 0.0

    with torch.no_grad():
        for labels, texts, offsets in dataloader:
            predicted_labels = model(texts, offsets)
            loss = criterion(predicted_labels, labels)

            total_loss += loss.item() * labels.size(0)
            total_acc += (predicted_labels.argmax(1) == labels).sum().item()
            total_count += labels.size(0)

    return total_loss / total_count, total_acc / total_count

# %% [markdown]
# ## 9. Bucle Principal de Entrenamiento
EPOCHS = 5
best_valid_loss = float('inf')

print("\n--- Iniciando Entrenamiento ---")

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()

    train_loss, train_acc = train_epoch(modelo, train_dataloader, criterion, optimizer)
    valid_loss, valid_acc = evaluate(modelo, valid_dataloader, criterion)

    scheduler.step()

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': modelo.state_dict(),
            'optimizer_state_dic  t': optimizer.state_dict(),
            'loss': valid_loss,
            'acc': valid_acc
        }
        torch.save(checkpoint, "checkpoint_mejor_modelo.pt")
        print(f"  --> ¡Nuevo mejor checkpoint guardado! (Val Loss: {best_valid_loss:.4f})")

    elapsed = time.time() - start_time
    print(f"Época {epoch:2d} | Tiempo: {elapsed:5.2f}s | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {valid_loss:.4f} | Val Acc: {valid_acc*100:.2f}%")

# %% [markdown]
# ## 10. Carga del Checkpoint y Evaluación en Test
print("\nCargando el mejor checkpoint registrado...")
checkpoint = torch.load("checkpoint_mejor_modelo.pt")
modelo.load_state_dict(checkpoint['model_state_dict'])

test_loss, test_acc = evaluate(modelo, test_dataloader, criterion)
print(f"\n--- Resultado Final en Test ---")
print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%")

# %% [markdown]
# ## 11. Función de Inferencia con `torch.compile`
DBPEDIA_LABELS = {
    1: 'Company', 2: 'EducationalInstitution', 3: 'Artist', 4: 'Athlete',
    5: 'OfficeHolder', 6: 'MeanOfTransportation', 7: 'Building',
    8: 'NaturalPlace', 9: 'Village', 10: 'Animal', 11: 'Plant',
    12: 'Album', 13: 'Film', 14: 'WrittenWork'
}

# Preparar modelo en CPU para inferencia rápida
modelo.to("cpu")
modelo.eval()

# Compilación previa única
compiled_model = torch.compile(modelo, mode="reduce-overhead")

def predict(texto, text_pipeline, model, labels_dict):
    with torch.no_grad():
        text_tensor = torch.tensor(text_pipeline(texto), dtype=torch.int64)
        offsets = torch.tensor([0])
        output = model(text_tensor, offsets)
        pred_idx = output.argmax(1).item() + 1
        return labels_dict.get(pred_idx, "Categoría desconocida")

# Pruebas de inferencia
ejemplo_1 = (
    "Nithari is a village in the western part of the state of Uttar Pradesh India "
    "bordering on New Delhi. Nithari forms part of the New Okhla Industrial "
    "Development Authority's planned industrial city Noida falling in Sector 31."
)
ejemplo_2 = "The Boeing 747 is a large, long-range wide-body airliner manufactured by Boeing Commercial Airplanes."

print("\n--- Resultados de Inferencia ---")
print(f"Ejemplo 1 -> Predicción: {predict(ejemplo_1, texto_pipeline, compiled_model, DBPEDIA_LABELS)}")
print(f"Ejemplo 2 -> Predicción: {predict(ejemplo_2, texto_pipeline, compiled_model, DBPEDIA_LABELS)}")
# %% [markdown]
# ## 12. Subida de Modelo a `Hugging face`
# Para subir el modelo a Hugging Face, primero debemos asegurarnos de tener instalada la librería `huggingface_hub`
!pip install huggingface_hub
from huggingface_hub import notebook_login
notebook_login()
# %% [markdown]
# ## 13. Creación de Repositorio y subida del modelo
from huggingface_hub import HfApi

api = HfApi()
api.create_repo(repo_id="jesusromerodev/clasificacion-DBpedia-jesus-romero", exist_ok=True)

!ls
# Salida esperada: checkpoint_mejor_modelo.pt  sample_data

api.upload_file(
    path_or_fileobj="./checkpoint_mejor_modelo.pt",
    path_in_repo="checkpoint_mejor_modelo.pt",
    repo_id="jesusromerodev/clasificacion-DBpedia-jesus-romero"
)
# %% [markdown]
# ## 14. Descarga de modelo
!mkdir weights

from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="jesusromerodev/clasificacion-DBpedia-jesus-romero",
    filename="checkpoint_mejor_modelo.pt",
    local_dir="weights/"
)

!ls weights
# Salida esperada: checkpoint_mejor_modelo.pt

# %%
torch.cuda.device_name(0)
# %%
