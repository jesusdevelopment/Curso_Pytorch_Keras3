# %% [markdown]
# ## Bibliotecas
!pip install --upgrade --force-reinstall fsspec datasets huggingface_hub
# %% [markdown]
# ##  Bibliotecas
!pip install torchtext
import time
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
# %% [markdown]
# ## Configuración de Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Usando el dispositivo: {device}')
# %% [markdown]
# ## Carga del Dataset
# Especificamos el namespace completo del dataset en Hugging Face
dataset = load_dataset("fancyzhx/dbpedia_14")
train_data = dataset["train"]
test_data = dataset["test"]
# %% [markdown]
# ## Exploración del Dataset
# 1. Ver qué splits/sets   existen
print("Splits disponibles:", dataset.keys())

# 2. Obtener la cantidad de componentes/filas en cada set
cant_train = len(dataset["train"])  # o dataset["train"].num_rows
cant_test = len(dataset["test"])    # o dataset["test"].num_rows

print(f"Train posee: {cant_train:,} ejemplos")
print(f"Test posee:  {cant_test:,} ejemplos")

# 3. Imprimir la estructura global del objeto
print(dataset)

# 4. Ver los nombres de los atributos/columnas
print("Columnas:", dataset["train"].column_names)
# Salida: ['label', 'title', 'content']

# 5. Ver la estructura detallada y tipos de datos (features)
print("Features:", dataset["train"].features)

# 6. Obtener el nombre textual de cada clase/etiqueta (0 a 13)
etiquetas = dataset["train"].features["label"].names
print("\nCategorías de clasificación (0 al 13):")
for idx, nombre in enumerate(etiquetas):
    print(f"  {idx}: {nombre}")
train_data=dataset["train"]
test_data=dataset["test"]

# %% [markdown]
# 7. Primer ejemplo
primer_ejemplo = dataset["train"][0]
print("\nAtributos del primer ejemplo:", list(primer_ejemplo.keys()))
print(f"Etiqueta (Label): {primer_ejemplo['label']}")
print(f"Título: {primer_ejemplo['title']}")
print(f"Texto: {primer_ejemplo['content'][:120]}...")
print("Ejemplo completo:")
print(primer_ejemplo)

# 8. Segundo ejemplo
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
    # Separa palabras y signos de puntuación, convirtiendo a minúsculas
    regex = re.compile(r"\w+|[^\w\s]")
    return lambda texto: regex.findall(texto.lower())

tokenizador = get_tokenizer()

class Vocab:
    def __init__(self, counter, specials=["<unk>"]):
        self.stoi = {}
        self.itos = []
        self.default_index = None
        
        # Insertar tokens especiales primero
        for spec in specials:
            self._add_token(spec)
            
        # Insertar palabras ordenadas por frecuencia
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

# Transformación de texto a lista de índices numéricos
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

# --- CORRECCIÓN AQUÍ ---
num_class = len(etiquetas)  # Se define la cantidad total de clases (14)
embedding_size = 100
vocab_size = len(vocab)

modelo = TextClassificationModel(
    vocab_size=vocab_size, 
    embed_dim=embedding_size, 
    num_class=num_class
).to(device)

print(f"\n¡Modelo instanciado correctamente para {num_class} clases!")
# --- FUNCIÓN UTILITARIA (OPCIONAL) ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Parámetros entrenables del modelo: {count_parameters(modelo):,}")
# %% [markdown]
# ## 7. Pérdida, Optimizador y DataLoaders
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(modelo.parameters(), lr=4.0)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False, collate_fn=collate_batch)

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
print("\n--- Iniciando Entrenamiento ---")

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    
    train_loss, train_acc = train_epoch(modelo, train_dataloader, criterion, optimizer)
    valid_loss, valid_acc = evaluate(modelo, test_dataloader, criterion)
    
    scheduler.step()
    elapsed = time.time() - start_time
    
    print(f"Época {epoch:2d} | Tiempo: {elapsed:5.2f}s | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {valid_loss:.4f} | Val Acc: {valid_acc*100:.2f}%")

# %% [markdown]
# ## 10. Función de Inferencia
def predecir_texto(texto, model, vocab, tokenizador, etiquetas, device):
    model.eval()
    with torch.no_grad():
        tokens = texto_pipeline(texto)
        text_tensor = torch.tensor(tokens, dtype=torch.int64).to(device)
        offsets = torch.tensor([0]).to(device)
        
        output = model(text_tensor, offsets)
        pred_idx = output.argmax(1).item()
        return etiquetas[pred_idx]

# Prueba de inferencia
ejemplo_prueba = "The Boeing 747 is a large, long-range wide-body airliner manufactured by Boeing Commercial Airplanes."
prediccion = predecir_texto(ejemplo_prueba, modelo, vocab, tokenizador, etiquetas, device)

print("\n--- Resultado de Inferencia ---")
print(f"Texto de entrada: '{ejemplo_prueba}'")
print(f"Clase predicha: {prediccion}")
# %%
