# %% [markdown]
# ## Bibliotecas
!pip install --upgrade --force-reinstall fsspec datasets huggingface_hub
# %%
!pip install torchtext
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


# Especificamos el namespace completo del dataset en Hugging Face
dataset = load_dataset("fancyzhx/dbpedia_14")

# Extraer el split de entrenamiento
train_data = dataset["train"]

# Obtener el primer registro
primer_ejemplo = train_data[0]

print("¡DBpedia cargado con éxito!")
print(f"Etiqueta (Label): {primer_ejemplo['label']}")
print(f"Título: {primer_ejemplo['title']}")
print(f"Texto: {primer_ejemplo['content'][:120]}...")
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
# Pipeline directo
texto_pipeline = lambda x: [vocab[token] for token in tokenizador(x)]

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
# %%
