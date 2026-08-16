 # %% [mark down]
## Importación de Bibliotecas y Configuración de Entorno
!pip install imagehash
!pip install --upgrade keras  # Aseguramos tener Keras 3+
!pip install torchmetrics -q

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile
from PIL import Image
import glob   
from io import BytesIO
import hashlib
from google.colab import drive
import datetime
import cv2
import random
import subprocess
import imagehash
import os# Para la detección de duplicados

# Importaciones de Machine Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from torchvision import transforms, datasets, models

# Métricas agrupadas usando la API pública principal
from torchmetrics import Accuracy, F1Score, ConfusionMatrix, MetricCollection

# %% [markdown]
## 1. Conexión con Google Drive y Carga de Datos

if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# Configuración de rutas
ruta_rar = "/content/drive/MyDrive/Data/Curso Prof TensorFlow/dataset_extraido.rar"
extract_dir = "/content/dataset_trabajo"

# Proceso de extracción con limpieza previa
if os.path.exists(ruta_rar):
    if os.path.exists(extract_dir):
        !rm -rf "{extract_dir}"
    
    os.makedirs(extract_dir, exist_ok=True)
    
    print("📦 Extrayendo dataset... Por favor, espera a que aparezca el mensaje de éxito.")
    !unrar x -o+ -idq "{ruta_rar}" "{extract_dir}/"
    
    print("🔍 Buscando carpetas de datos...")
    hallazgos = glob.glob(os.path.join(extract_dir, "**/Training"), recursive=True)
    
    if hallazgos:
        base_real = os.path.dirname(hallazgos[0])
        train_dir = os.path.join(base_real, 'Training')
        test_dir = os.path.join(base_real, 'Testing')
        
        print(f"✅ ¡Éxito! Carpetas encontradas en: {base_real}")
        print(f"📍 Train: {train_dir}")
        print(f"📍 Test:  {test_dir}")
        
        extract_dir = base_real 
    else:
        print("❌ ERROR: El .rar se extrajo pero no contiene una carpeta llamada 'Training'.")
        !ls -R "{extract_dir}" | head -n 10
else:
    print(f"❌ ERROR: No se encontró el archivo en Drive: {ruta_rar}")

# %% [markdown]
## 2. EDA: Balanceo, Limpieza y Análisis Dimensional

print("\n--- INICIANDO EDA Y LIMPIEZA DE DATOS ---")

# 5.1 Balanceo de Clases
sets = ['Training', 'Testing']
MIS_CLASES_BRAIN = ['glioma', 'meningioma', 'notumor', 'pituitary']
stats = []

for dataset_type in sets:
    set_path = os.path.join(extract_dir, dataset_type)
    if not os.path.exists(set_path):
        print(f"⚠️ Alerta: No se encontró la carpeta: {set_path}")
        continue
    for label in MIS_CLASES_BRAIN:
        path = os.path.join(set_path, label)
        if os.path.exists(path):
            count = len(os.listdir(path))
            stats.append({'label': label, 'count': count, 'dataset': dataset_type})
        else:
            print(f"❌ Carpeta de clase no encontrada: {path}")

df_stats = pd.DataFrame(stats)

if df_stats.empty:
    print("FATAL: El DataFrame está vacío. Revisa las rutas de 'extract_dir'.")
else:
    plt.figure(figsize=(12, 6))
    sns.barplot(x='label', y='count', hue='dataset', data=df_stats)
    plt.title('Distribución de Clases: Glioma, Meningioma, No Tumor y Pituitaria')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# 5.2 Detección de archivos Corruptos
def check_images(directory):
    print("\nBuscando imágenes corruptas...")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path)
                    img.verify() 
                except (IOError, SyntaxError):
                    print(f'Archivo corrupto eliminado: {path}')
                    os.remove(path)

check_images(extract_dir)

# 5.3 Detección de Archivos Duplicados (Hashing)
def eliminar_duplicados_visuales(directorio):
    print(f"\nBuscando duplicados visuales en: {directorio}")
    hashes_vistos = {}
    duplicados_eliminados = 0
    
    for root, dirs, files in os.walk(directorio):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, file)
                try:
                    with Image.open(path) as img:
                        v_hash = imagehash.dhash(img)
                    if v_hash in hashes_vistos:
                        print(f"Eliminando duplicado visual: {path}")
                        os.remove(path)
                        duplicados_eliminados += 1
                    else:
                        hashes_vistos[v_hash] = path
                except Exception as e:
                    print(f"No se pudo procesar {file}: {e}")
                
    print(f"¡Limpieza terminada! Se eliminaron {duplicados_eliminados} archivos duplicados.")

eliminar_duplicados_visuales(train_dir)
eliminar_duplicados_visuales(test_dir)

# 5.4 Análisis Dimensional (Tamaños y Proporciones)
print("\nAnalizando dimensiones de las imágenes...")
sets_to_analyze = ['Training', 'Testing']
colors = {'Training': 'blue', 'Testing': 'orange'}
plt.figure(figsize=(10, 6))

widths, heights = [], [] # Para el resumen numérico global

for dataset_type in sets_to_analyze:
    w_temp, h_temp = [], []
    current_path = os.path.join(extract_dir, dataset_type)
    for root, dirs, files in os.walk(current_path):
        for file in files:
            if file.endswith('.jpg'):
                try:
                    with Image.open(os.path.join(root, file)) as im:
                        w, h = im.size
                        w_temp.append(w)
                        h_temp.append(h)
                        widths.append(w)
                        heights.append(h)
                except Exception as e:
                    pass
    plt.scatter(w_temp, h_temp, alpha=0.3, label=dataset_type, color=colors[dataset_type])

plt.xlabel('Ancho (píxeles)')
plt.ylabel('Alto (píxeles)')
plt.title('Comparación de Dimensiones: Entrenamiento vs. Test')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_test = sum([len(files) for r, d, files in os.walk(test_dir)])

print("--- RESUMEN DEL DATASET ---")
if widths and heights:
    print(f"Límite máximo de dimensiones: {max(widths)} (ancho) x {max(heights)} (alto)")
    print(f"Límite mínimo de dimensiones: {min(widths)} (ancho) x {min(heights)} (alto)")
print(f"Total de imágenes para entrenamiento: {total_train}")
print(f"Total de imágenes para prueba (Test): {total_test}")
print(f"Proporción de entrenamiento: {total_train / (total_train + total_test):.2f}")
print(f"Proporción de prueba: {total_test / (total_train + total_test):.2f}")

# 5.5 Análisis de Intensidad de Píxeles
# 5.5 Análisis de Intensidad de Píxeles (Corregido)
def analizar_intensidades(extract_dir, sets=['Training', 'Testing'], sample_size=300):
    plt.figure(figsize=(12, 6))
    colores = {'Training': 'blue', 'Testing': 'orange'}
    print("\nIniciando análisis de intensidades de píxel...")

    for dataset_type in sets:
        dataset_path = os.path.join(extract_dir, dataset_type)
        all_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                # Permite imágenes .png, .jpeg y .jpg
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    all_files.append(os.path.join(root, file))
        
        if not all_files:
            print(f"⚠️ No se encontraron imágenes en: {dataset_path}")
            continue

        sampled_files = random.sample(all_files, min(sample_size, len(all_files)))
        hist_acumulado = np.zeros(256) # Array 1D de forma (256,)
        
        for img_path in sampled_files:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # .ravel() aplana el resultado a un vector 1D de (256,)
                hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
                hist_acumulado += hist
        
        total_pixels = hist_acumulado.sum()
        if total_pixels > 0:
            hist_acumulado /= total_pixels

        plt.plot(hist_acumulado, color=colores[dataset_type], label=f'{dataset_type} (n={len(sampled_files)})', alpha=0.8)

    plt.title('Distribución de Intensidades de Píxel: Training vs. Testing')
    plt.xlabel('Intensidad (0 = Negro absoluto, 255 = Blanco absoluto)')
    plt.ylabel('Frecuencia Relativa')
    plt.xlim([0, 256])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

analizar_intensidades(extract_dir)

# %%[markdown]
# ## 3. Carga de Datos

# ------------------------------------------------------------------
# 1. DEFINICIÓN DE TRANSFORMACIONES
# ------------------------------------------------------------------
# Data Augmentation solo para entrenamiento
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Para Validación y Test solo redimensionamos y normalizamos
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ------------------------------------------------------------------
# 2. CARGA DE LOS DATASETS (Con Split 80/20 para Validación)
# ------------------------------------------------------------------
# Cargamos la misma carpeta 'train' DOS VECES con transformaciones diferentes
dataset_para_train = datasets.ImageFolder(root=train_dir, transform=train_transforms)
dataset_para_val   = datasets.ImageFolder(root=train_dir, transform=val_test_transforms)

# Calculamos los tamaños para el split (80% train, 20% val)
total_size = len(dataset_para_train)
val_size = int(0.2 * total_size)
train_size = total_size - val_size

# Hacemos el split aleatorio. Usamos un generador con semilla para reproducibilidad
generador = torch.Generator().manual_seed(42)
train_subset_temp, val_subset_temp = random_split(dataset_para_train, [train_size, val_size], generator=generador)

# EL TRUCO ESTÁ AQUÍ:
# train_subset_temp tiene las transformaciones de train (¡Correcto!)
train_dataset = train_subset_temp 

# Pero val_subset_temp también tiene las de train. Lo que hacemos es extraer 
# los índices que PyTorch eligió aleatoriamente y se los aplicamos a 'dataset_para_val'
val_dataset = Subset(dataset_para_val, val_subset_temp.indices)

# El de test se carga normal desde su carpeta
test_dataset = datasets.ImageFolder(root=test_dir, transform=val_test_transforms)

num_clases = len(dataset_para_train.classes)
print(f"Clases detectadas: {dataset_para_train.classes} ({num_clases} clases)")
print(f"Imágenes para entrenamiento: {len(train_dataset)}")
print(f"Imágenes para validación: {len(val_dataset)}")
print(f"Imágenes para test: {len(test_dataset)}")

# ------------------------------------------------------------------
# 3. CREACIÓN DE LOS DATALOADERS
# ------------------------------------------------------------------
# Cálculo del paralelismo óptimo
num_workers = min(8, os.cpu_count() or 1)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

# %% [markdown]
# ## 4. Arquitectura de la CNN
# Definición de la arquitectura CNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Classificador_Lesiones(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        ).to(device)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Linear(64, 128), # Ajustar dimensiones según el tamaño de entrada
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes).to(device)
        )
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x