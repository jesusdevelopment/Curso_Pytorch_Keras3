# %% [markdown]
# ## Bibliotecas
!pip install --upgrade --force-reinstall fsspec datasets huggingface_hub
# %% [markdown]
# ##  Carga del DataSet
!pip install torchtext
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

# Especificamos el namespace completo del dataset en Hugging Face
dataset = load_dataset("fancyzhx/dbpedia_14")

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
# ## Tokenización y Procesamiento de Datos
import re
from collections import Counter

# 1. Tokenizador equivalente a "basic_english" de torchtext
def get_tokenizer():
    # Separa palabras y signos de puntuación, convirtiendo a minúsculas
    regex = re.compile(r"\w+|[^\w\s]")
    return lambda texto: regex.findall(texto.lower())

tokenizador = get_tokenizer()

# 2. Clase que emula el objeto Vocab de torchtext
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
        # Si la palabra no existe en el vocabulario, devuelve el índice por defecto (<unk>)
        return self.stoi.get(token, self.default_index)

    def __len__(self):
        return len(self.itos)

# 3. Construir el vocabulario desde el dataset de Hugging Face
print("Procesando tokens y construyendo vocabulario...")
counter = Counter()

for example in train_data:
    texto = f"{example['title']} {example['content']}"
    counter.update(tokenizador(texto))

vocab = Vocab(counter, specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

print(f"¡Vocabulario creado con éxito! Tamaño total: {len(vocab):,} palabras.")
# %%
# Tokenizar
print(tokenizador("Hello how are you? I am a platzi student"))
print(tokenizador("This is a test sentence"))
# Pipeline directo
texto_pipeline = lambda x: [vocab[token] for token in tokenizador(x)]
label_pipeline = lambda x: int(x) - 1
# Prueba
print(texto_pipeline("Hello how are you? I am a platzi student"))
print(texto_pipeline("This is a test sentence"))
# %% [markdown]
# 4. Preparación de los datos para el DataLoader


from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_batch(batch):
    label_list, text_list = [], []
    offsets = [0]
    
    for item in batch:
        # 1. Extraer etiqueta y ajustar si es necesario
        label_list.append(item['label'])
        
        # 2. Combinar título y contenido para tener el texto completo
        texto_completo = f"{item['title']} {item['content']}"
        
        # 3. Procesar texto a tensores de índices
        processed_text = torch.tensor(texto_pipeline(texto_completo), dtype=torch.int64)
        text_list.append(processed_text)
        
        # 4. Guardar longitud del texto para calcular los offsets
        offsets.append(processed_text.size(0))
        
    # Convertir listas a tensores de PyTorch
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    
    # Enviar tensores a la GPU/CPU (corregido typo 'lable_list' -> 'label_list')
    return label_list.to(device), text_list.to(device), offsets.to(device)

# Usamos directamente train_data de Hugging Face
dataloader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=collate_batch)

# Probar que el DataLoader funciona correctamente consumiendo un batch
labels, texts, offsets = next(iter(dataloader))
print("¡DataLoader configurado con éxito!")
print("Shape de Etiquetas (Labels):", labels.shape)
print("Shape de Textos concatenados:", texts.shape)
print("Shape de Offsets:", offsets.shape)
dataloader
# %% [markdown]
# 6. Creación del Modelo con nn.Module
# Definición de la clase del modelo para clasificación de texto
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset  # <--- Importar Hugging Face Datasets

# ------------------------------------------------------------------
# Paso previo: Cargar el dataset para definir 'train_data'
# ------------------------------------------------------------------
dataset = load_dataset("fancyzhx/dbpedia_14")
train_data = dataset["train"]

# ------------------------------------------------------------------
# Definición del Modelo
# ------------------------------------------------------------------
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
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

# 1. Extraer número de clases únicas desde el dataset de Hugging Face
num_class = len(set(train_data["label"]))

# 2. Configuración de dimensiones
vocab_size = len(vocab)
embedding_size = 100

# 3. Instanciación del modelo y envío al dispositivo
modelo = TextClassificationModel(
    vocab_size=vocab_size, 
    embed_dim=embedding_size, 
    num_class=num_class
).to(device)

print("¡Modelo instanciado correctamente!")
print(f"• Clases a clasificar: {num_class}")
print(f"• Tamaño del vocabulario: {vocab_size:,}")
print(f"• Arquitectura en {device}:\n", modelo)# %% [markdown]
4 # %%
# 7. Inicialización de pesos
 def init_weights(self):
     initrange = 0.5
     self.embedding.weight.data.uniform_(-initrange, initrange)
     self.fc.weight.data.uniform_(-initrange, initrange)
     self.fc.bias.data.zero_()
     # %% [markdown]
     # 8. Configuración del dispositivo y entrenamiento
     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     # %% [markdown]
     # 9. Definición de la función de pérdida y optimizador
     criterion = torch.nn.CrossEntropyLoss()
     optimizer = torch.optim.SGD(modelo.parameters(), lr=4.0)
     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
     
     

     

