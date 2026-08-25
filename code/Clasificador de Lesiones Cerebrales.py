# %% [markdown]
## 1. Library Imports
!pip install imagehash -q
!pip install torchmetrics -q
!pip install grad_cam -q

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
import copy
import os
import time
from typing import Optional, Tuple, Dict, List  # <-- AÑADIDO: Tipos para type hints

# Machine Learning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset, random_split
from torchvision import transforms, datasets, models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc, 
    roc_auc_score
)
from sklearn.preprocessing import label_binarize


from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix, MetricCollection
# %% [markdown]
## 2. Google Drive 

if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

# 2.1 Paths
ruta_rar = "/content/drive/MyDrive/Data/Curso Prof TensorFlow/dataset_extraido.rar"
extract_dir = "/content/dataset_trabajo"

# 2.2
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
## 3. EDA

print("\n--- INICIANDO EDA Y LIMPIEZA DE DATOS ---")

# 3.1 Classes Balance
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

# 3.2 Corrupts Files detection
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

# 3.3 Hashing
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

# 3.4 
print("\nAnalizando dimensiones de las imágenes...")
sets_to_analyze = ['Training', 'Testing']
colors = {'Training': 'blue', 'Testing': 'orange'}
plt.figure(figsize=(10, 6))

widths, heights = [], [] 

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


# 3.5 Análisis de Intensidad de Píxeles (Corregido)
def analizar_intensidades(extract_dir, sets=['Training', 'Testing'], sample_size=300):
    plt.figure(figsize=(12, 6))
    colores = {'Training': 'blue', 'Testing': 'orange'}
    print("\nIniciando análisis de intensidades de píxel...")

    for dataset_type in sets:
        dataset_path = os.path.join(extract_dir, dataset_type)
        all_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
              
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
## 4. Data Loading

# ------------------------------------------------------------------
# 4.1 DEFINICIÓN DE TRANSFORMACIONES
# ------------------------------------------------------------------
# Data Augmentation solo para entrenamiento
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
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
# Para visualización, definimos una transformación sin normalización para mantener los valores de píxeles originales:
vis_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()])

# ------------------------------------------------------------------
# 4.2 CARGA DE LOS DATASETS (Con Split 80/20 para Validación)
# ------------------------------------------------------------------
# Cargamos la misma carpeta 'train' DOS VECES con transformaciones diferentes
dataset_para_train = datasets.ImageFolder(root=train_dir, transform=train_transforms)
dataset_para_val   = datasets.ImageFolder(root=train_dir, transform=val_test_transforms)
train_dataset_image=  datasets.ImageFolder(root=train_dir, transform=vis_transforms)
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

num_classes = len(dataset_para_train.classes)
print(f"Clases detectadas: {dataset_para_train.classes} ({num_classes} clases)")
print(f"Imágenes para entrenamiento: {len(train_dataset)}")
print(f"Imágenes para validación: {len(val_dataset)}")
print(f"Imágenes para test: {len(test_dataset)}")

# ------------------------------------------------------------------
# 4.3 CREACIÓN DE LOS DATALOADERS
# ------------------------------------------------------------------
# Cálculo del paralelismo óptimo
num_workers = min(8, os.cpu_count() or 1)

image_loader = DataLoader(train_dataset_image, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
# %% [markdown]
## Visualización de 5 muestras iniciales

images, labels = next(iter(image_loader))
plt.figure(figsize=(12, 12))
for i in range(5):
    ax = plt.subplot(1, 5, i + 1)
    
    # Permutar de [C, H, W] a [H, W, C] para Matplotlib
    plt.imshow(images[i].permute(1, 2, 0).numpy())
    
    # Obtener el nombre de la clase usando el índice numérico del label
    # Opción A (Usando la lista que ya tienes definida):
    nombre_clase = MIS_CLASES_BRAIN[labels[i].item()]
    
    # Opción B (Usando la propiedad nativa del dataset de PyTorch):
    # nombre_clase = train_dataset_image.classes[labels[i].item()]
    
    plt.title(f"Clase: {nombre_clase}")
    plt.axis("off")

plt.show()
# %% [markdown]
# %% [markdown]
## 5. Arquitectura de la CNN
# Definición de la arquitectura CNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Classificador_Manual(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d(output_size=(2, 2)),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            # Al usar AdaptiveAvgPool2d(2, 2), la salida plana es canales * alto * ancho
            nn.Linear(128 * 2 * 2, 256), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x 

class MultiModeloTransfer(nn.Module):
    def __init__(self, nombre_modelo, num_classes):
        super().__init__()
        self.nombre_modelo = nombre_modelo
        
        if self.nombre_modelo == 'densenet':
            self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
            self._congelar_pesos()
            num_ftrs = self.backbone.classifier.in_features
            self.backbone.classifier = self._crear_cabeza_clasificacion(num_ftrs, num_classes)
            
        elif self.nombre_modelo == 'resnet':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            self._congelar_pesos()
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = self._crear_cabeza_clasificacion(num_ftrs, num_classes)
            
        elif self.nombre_modelo == 'efficientnet':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self._congelar_pesos()
            num_ftrs = self.backbone.classifier[1].in_features
            self.backbone.classifier = self._crear_cabeza_clasificacion(num_ftrs, num_classes, es_efficientnet=True)

    def _congelar_pesos(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _crear_cabeza_clasificacion(self, num_in_features, num_classes, es_efficientnet=False):
        if es_efficientnet:
            return nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(num_in_features, num_classes)
            )
        else:
            return nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_in_features, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )

    def forward(self, x):
        return self.backbone(x)

    # ------------------------------------------------------------------
    # NUEVO MÉTODOS: DESCONGELAMIENTO ESPECÍFICO DE BLOQUES
    # ------------------------------------------------------------------
    def descongelar_ultimos_bloques(self):
        """
        Descongelación orientada de los últimos bloques convolucionales
        según la topología específica de cada red.
        """
        if self.nombre_modelo == 'resnet':
            # ResNet50 se divide en layer1, layer2, layer3, layer4.
            # Descongelamos los dos/tres últimos bloques principales (layer3 y layer4)
            bloques_a_descongelar = [self.backbone.layer3, self.backbone.layer4]
            
        elif self.nombre_modelo == 'densenet':
            # DenseNet121 se compone de denseblock1..4 y transition1..3 en .features
            # Descongelamos denseblock3, transition3, denseblock4 y norm5
            bloques_a_descongelar = [
                self.backbone.features.denseblock3,
                self.backbone.features.transition3,
                self.backbone.features.denseblock4,
                self.backbone.features.norm5
            ]
            
        elif self.nombre_modelo == 'efficientnet':
            # EfficientNet-B0 tiene 9 bloques secuenciales en .features (índices 0 a 8)
            # Descongelamos los últimos 3 bloques (índices 6, 7 y 8)
            bloques_a_descongelar = [self.backbone.features[6:]]

        # Aplicar el cambio de estado a los parámetros seleccionados
        for bloque in bloques_a_descongelar:
            for param in bloque.parameters():
                param.requires_grad = True

        # Obtener y retornar parámetros divididos para el optimizador
        params_backbone_descongelados = []
        for bloque in bloques_a_descongelar:
            params_backbone_descongelados.extend([p for p in bloque.parameters() if p.requires_grad])
            
        if self.nombre_modelo == 'resnet':
            params_head = [p for p in self.backbone.fc.parameters() if p.requires_grad]
        else:
            params_head = [p for p in self.backbone.classifier.parameters() if p.requires_grad]

        return params_backbone_descongelados, params_head
    
# Instanciamos los modelos
modelo_denso = MultiModeloTransfer('densenet', num_classes).to(device)
modelo_residual = MultiModeloTransfer('resnet', num_classes).to(device)
modelo_eficiente = MultiModeloTransfer('efficientnet', num_classes).to(device)
modelo_manual = Classificador_Manual(num_classes=num_classes).to(device)

print(modelo_manual)

# %% [markdown]
## 6. Función de Pérdida, Optimizador y Scheduler

# 6.1. Extraemos las etiquetas reales del Subset de entrenamiento
etiquetas_train = [train_dataset.dataset.targets[i] for i in train_dataset.indices]

# 6.2. Convertimos a tensor y contamos las frecuencias automáticamente
etiquetas_tensor = torch.tensor(etiquetas_train)
class_counts = torch.bincount(etiquetas_tensor).float()

# 6.3. Tu lógica de pesos inversamente proporcionales
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()  # Normalización opcional

print(f"Frecuencias calculadas: {class_counts.tolist()}")
print(f"Pesos asignados: {class_weights.tolist()}")

# 6.4. Función de pérdida
criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device),
    label_smoothing=0.1  # Regularización para evitar probabilidades extremas
)
# 6.5. Optimizador y Scheduler
# %% [markdown]
## 7. Bucle de Entrenamiento, Validación y Registro de Métricas

# --- CLASE EARLY STOPPING ---
class EarlyStopping:
    def __init__(self, patience: int = 5, path: str = "checkpoint.pt"):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, model: nn.Module):
        torch.save(model.state_dict(), self.path)


# --- CLASE ENGINE TRAINER (POO OPTIMIZADA) ---
class EngineTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        max_grad_norm: float = 1.0,
        use_amp: bool = True,
        scheduler_step_per_batch: bool = False,
        checkpoint_path: str = "best_checkpoint.pt"
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scheduler_step_per_batch = scheduler_step_per_batch
        self.checkpoint_path = checkpoint_path
        
        # Escalador para Automatic Mixed Precision
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

    def _train_one_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for images, targets in self.train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()

            if self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None and self.scheduler_step_per_batch:
                self.scheduler.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            correct_predictions += torch.sum(preds == targets).item()
            total_samples += images.size(0)

        return running_loss / total_samples, correct_predictions / total_samples

    def _evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.inference_mode():
            for images, targets in data_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item() * images.size(0)
                preds = torch.argmax(outputs, dim=1)
                correct_predictions += torch.sum(preds == targets).item()
                total_samples += images.size(0)

        return running_loss / total_samples, correct_predictions / total_samples

    def fit(self, epochs: int, early_stopping_patience: int = 5) -> Dict[str, list]:
        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        early_stopping = EarlyStopping(patience=early_stopping_patience, path=self.checkpoint_path)

        print(f"--- Entrenando en: {self.device} | AMP: {self.use_amp} | Checkpoint: {self.checkpoint_path} ---")
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            train_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc = self._evaluate(self.val_loader)

            if self.scheduler is not None and not self.scheduler_step_per_batch:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            current_lr = self.optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            print(
                f"Época [{epoch+1:02d}/{epochs:02d}] ({epoch_time:.2f}s) | "
                f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            early_stopping(val_loss, self.model)
            if early_stopping.early_stop:
                print(f"\nEarly Stopping activado en época {epoch+1}.")
                break

        total_time = time.time() - start_time
        print(f"--- Entrenamiento Completado en {total_time:.2f}s ---")
        
        self.model.load_state_dict(torch.load(self.checkpoint_path))
        return history

    def get_predictions(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extrae las etiquetas reales, predicciones y probabilidades softmax del conjunto dado."""
        self.model.eval()
        all_targets, all_preds, all_probs = [], [], []

        with torch.inference_mode():
            for images, targets in data_loader:
                images = images.to(self.device, non_blocking=True)
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)
                    probs = torch.softmax(outputs, dim=1)

                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return np.array(all_targets), np.array(all_preds), np.array(all_probs)


# %% [markdown]
## 8. Funciones Modulares de Evaluación y Métricas Visuales

def graficar_historias(historias: Dict[str, dict]):
    """Grafica la evolución de Loss y Accuracy en entrenamiento vs validación."""
    fig, axes = plt.subplots(len(historias), 2, figsize=(14, 4 * len(historias)))
    if len(historias) == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (nombre, hist) in enumerate(historias.items()):
        # Gráfica de Pérdida
        axes[idx, 0].plot(hist['train_loss'], label='Train Loss', color='blue')
        axes[idx, 0].plot(hist['val_loss'], label='Val Loss', color='orange')
        axes[idx, 0].set_title(f'Pérdida - {nombre}')
        axes[idx, 0].set_xlabel('Época')
        axes[idx, 0].set_ylabel('Loss')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True, linestyle='--', alpha=0.5)

        # Gráfica de Exactitud
        axes[idx, 1].plot(hist['train_acc'], label='Train Acc', color='blue')
        axes[idx, 1].plot(hist['val_acc'], label='Val Acc', color='orange')
        axes[idx, 1].set_title(f'Exactitud (Accuracy) - {nombre}')
        axes[idx, 1].set_xlabel('Época')
        axes[idx, 1].set_ylabel('Accuracy')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def evaluar_modelos_completo(
    trainers_dict: Dict[str, EngineTrainer],
    test_loader: DataLoader,
    clases: List[str]
):
    """
    Calcula e imprime el informe completo de métricas (Precision, Recall, F1, Accuracy),
    matrices de confusión y Curvas ROC-AUC Multiclase (One-vs-Rest).
    """
    num_clases = len(clases)
    fig_cm, axes_cm = plt.subplots(1, len(trainers_dict), figsize=(5 * len(trainers_dict), 4.5))
    fig_roc, axes_roc = plt.subplots(1, len(trainers_dict), figsize=(5.5 * len(trainers_dict), 4.5))

    if len(trainers_dict) == 1:
        axes_cm = [axes_cm]
        axes_roc = [axes_roc]

    resumen_metricas = []

    for idx, (nombre, trainer) in enumerate(trainers_dict.items()):
        # 8.1 Extraer predicciones y probabilidades softmax
        y_true, y_pred, y_probs = trainer.get_predictions(test_loader)

        # 8.2 Imprimir reporte detallado de clasificación (Scikit-Learn)
        print(f"\n==================== REPORTE DE EVALUACIÓN: {nombre.upper()} ====================")
        report_dict = classification_report(y_true, y_pred, target_names=clases, output_dict=True)
        print(classification_report(y_true, y_pred, target_names=clases, digits=4))

        # 8.3 Guardar resumen macro para tabla comparativa
        resumen_metricas.append({
            'Modelo': nombre,
            'Accuracy': report_dict['accuracy'],
            'Precision (Macro)': report_dict['macro avg']['precision'],
            'Recall (Macro)': report_dict['macro avg']['recall'],
            'F1-Score (Macro)': report_dict['macro avg']['f1-score'],
            'ROC-AUC (Macro OvR)': roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        })

        # 8.4 Matriz de Confusión
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes_cm[idx],
                    xticklabels=clases, yticklabels=clases)
        axes_cm[idx].set_title(f'Matriz Confusión: {nombre}')
        axes_cm[idx].set_xlabel('Predicción')
        axes_cm[idx].set_ylabel('Clase Real')

        # 8.5. Curvas ROC-AUC Multiclase (One-vs-Rest)
        y_true_bin = label_binarize(y_true, classes=list(range(num_clases)))
        
        for c_idx in range(num_clases):
            fpr, tpr, _ = roc_curve(y_true_bin[:, c_idx], y_probs[:, c_idx])
            roc_auc = auc(fpr, tpr)
            axes_roc[idx].plot(fpr, tpr, label=f'{clases[c_idx]} (AUC = {roc_auc:.3f})')

        axes_roc[idx].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes_roc[idx].set_xlim([0.0, 1.0])
        axes_roc[idx].set_ylim([0.0, 1.05])
        axes_roc[idx].set_xlabel('Tasa de Falsos Positivos (FPR)')
        axes_roc[idx].set_ylabel('Tasa de Verdaderos Positivos (TPR)')
        axes_roc[idx].set_title(f'Curvas ROC (OvR): {nombre}')
        axes_roc[idx].legend(loc="lower right")
        axes_roc[idx].grid(True, linestyle='--', alpha=0.5)

    fig_cm.tight_layout()
    fig_roc.tight_layout()
    plt.show()

    # Mostrar dataframe con la comparación global de métricas
    df_resumen = pd.DataFrame(resumen_metricas)
    print("\n==================== RESUMEN COMPARATIVO DE MÉTRICAS ====================")
    print(df_resumen.to_string(index=False))
    return df_resumen


# %% [markdown]
## 9. Ejecución Completa del Experimento

# 9.1 Definición de modelos a evaluar
diccionario_modelos = {
   # 'CNN_Manual': modelo_manual,
    'DenseNet121': modelo_denso,
    'ResNet50': modelo_residual,
    'EfficientNetB0': modelo_eficiente
}

historias = {}
trainers= {} 

for nombre, mod in diccionario_modelos.items():
    print(f"\n---> Configurando Fine-Tuning para: {nombre}")
    
    if isinstance(mod, MultiModeloTransfer):
        # 1. Descongela bloques específicos (DenseNet, ResNet o EfficientNet)
        params_backbone, params_head = mod.descongelar_ultimos_bloques()
         
        # 2. Asigna LR diferenciado: bajo para el backbone, más alto para la cabeza
        optimizer_model = torch.optim.AdamW([
            {'params': params_backbone, 'lr': 1e-5},
            {'params': params_head,     'lr': 1e-3}
        ], weight_decay=1e-2)
    else:
        # CNN_Manual optimiza todas sus capas con LR estándar
        optimizer_model = torch.optim.AdamW(
            mod.parameters(), 
            lr=1e-3, 
            weight_decay=1e-2
        )

    scheduler_model = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_model, mode='min', patience=2, factor=0.5
    )

    trainer = EngineTrainer(
        model=mod,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer_model,
        scheduler=scheduler_model,
        use_amp=True,
        checkpoint_path=f"best_{nombre}_ft.pt"
    )

    historia = trainer.fit(epochs=15, early_stopping_patience=3)
    
    trainers[nombre] = trainer
    historias[nombre] = historia

# 3. Visualización de la evolución (Loss y Accuracy)
graficar_historias(historias)

# 4. Evaluación exhaustiva en el Test Set (Métricas, Matriz de Confusión y ROC-AUC)
df_metricas_finales = evaluar_modelos_completo(
    trainers_dict=trainers,
    test_loader=test_loader,
    clases=MIS_CLASES_BRAIN
)
# %%[markdown]
# ##. Grand-Cam

# %% [markdown]
# ## 8. Visualización con Grad-CAM
# Para visualizar las regiones de interés en las imágenes, utilizaremos Grad-CAM.
# Esto nos permite entender qué partes de la imagen están activando la predicción del modelo.
# Importamos las librerías necesarias para Grad-CAM

def evaluar_por_categorias_gradcam_seguro(trainers_dict, test_loader, clases):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Configurar los objetos Grad-CAM para cada modelo
    cams = {}
    for nombre_modelo, trainer in trainers_dict.items():
        modelo_wrapper = trainer.model
        modelo_wrapper.to(device)
        modelo_wrapper.eval()
        red_interna = modelo_wrapper.backbone
        
        if "ResNet" in nombre_modelo:
            target_layers = [red_interna.layer4[-1]]
        elif "DenseNet" in nombre_modelo or "EfficientNet" in nombre_modelo:
            target_layers = [red_interna.features[-1]]
        else:
            raise ValueError(f"Arquitectura {nombre_modelo} no soportada.")
            
        cams[nombre_modelo] = GradCAM(model=modelo_wrapper, target_layers=target_layers)

    # 2. Búsqueda robusta de una muestra por cada categoría disponible
    muestras_por_clase = {}
    clases_objetivo = set(range(len(clases)))
    
    for images, labels in test_loader:
        for img, label in zip(images, labels):
            clase_idx = label.item()
            if clase_idx in clases_objetivo and clase_idx not in muestras_por_clase:
                muestras_por_clase[clase_idx] = img
            if len(muestras_por_clase) == len(clases):
                break
        if len(muestras_por_clase) == len(clases):
            break

    print(f"✅ Se encontraron muestras para las clases: {[clases[k] for k in muestras_por_clase.keys()]}")

    # 3. Iterar sobre cada categoría encontrada y generar su comparativa
    for clase_idx, input_tensor in muestras_por_clase.items():
        input_tensor = input_tensor.unsqueeze(0).to(device)
        nombre_clase_real = clases[clase_idx]
        
        print(f"\n------------------------------------------")
        print(f" Evaluando Categoría: {nombre_clase_real}")
        print(f"------------------------------------------")
        
        num_columnas = len(trainers_dict) + 1
        fig, axes = plt.subplots(1, num_columnas, figsize=(4 * num_columnas, 5))
        
        # Columna 0: Imagen Original Limpia
        img_original = input_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_original = np.clip(std * img_original + mean, 0, 1)
        
        axes[0].imshow(img_original)
        axes[0].set_title(f"Imagen Original\nReal: {nombre_clase_real}", color='blue', fontweight='bold', fontsize=11)
        axes[0].axis('off')
        
        # Columnas Restantes: Modelos con Grad-CAM
        for idx, (nombre_modelo, trainer) in enumerate(trainers_dict.items()):
            modelo_wrapper = trainer.model
            
            with torch.no_grad():
                salida = modelo_wrapper(input_tensor)
                pred_idx = salida.argmax(dim=1).item()
                
            targets = [ClassifierOutputTarget(pred_idx)]
            grayscale_cam = cams[nombre_modelo](input_tensor=input_tensor, targets=targets)[0, :]
            
            visualization = show_cam_on_image(img_original, grayscale_cam, use_rgb=True)
            
            ax = axes[idx + 1]
            ax.imshow(visualization)
            
            color_texto = "green" if clase_idx == pred_idx else "red"
            ax.set_title(f"{nombre_modelo}\nPred: {clases[pred_idx]}", color=color_texto, fontweight='bold', fontsize=11)
            ax.axis('off')
            
        plt.suptitle(f"Evaluación de Lesión: {nombre_clase_real.upper()}", fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

# --- EJECUCIÓN ---
evaluar_por_categorias_gradcam_seguro(trainers, test_loader, MIS_CLASES_BRAIN)
 
# %% [markdown]
# ##. Inferencia

def predecir_y_mostrar_imagen(ruta_imagen, nombre_modelo, trainer_dict, clases, transformacion_val):
    """
    Realiza inferencia sobre una única imagen nueva, muestra la imagen gráficamente 
    y desglosa las probabilidades del modelo.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Recuperar el modelo del diccionario de trainers y ponerlo en modo evaluación
    if nombre_modelo not in trainer_dict:
        raise ValueError(f"El modelo '{nombre_modelo}' no se encuentra en el diccionario de trainers.")
        
    trainer = trainer_dict[nombre_modelo]
    modelo = trainer.model
    modelo.to(device)
    modelo.eval()
    
    # Cargar los mejores pesos guardados en disco si existen
    checkpoint_path = f"best_{nombre_modelo}_ft.pt"
    if os.path.exists(checkpoint_path):
        modelo.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # 2. Cargar la imagen original (para mostrarla) y preprocesar la versión para el modelo
    if not os.path.exists(ruta_imagen):
        raise FileNotFoundError(f"No se encontró la imagen en la ruta: {ruta_imagen}")
        
    imagen_pil = Image.open(ruta_imagen).convert("RGB")
    tensor_imagen = transformacion_val(imagen_pil).unsqueeze(0).to(device)
    
    # 3. Inferencia (Forward Pass sin gradientes)
    with torch.no_grad():
        outputs = modelo(tensor_imagen)
        probabilidades = F.softmax(outputs, dim=1)[0]
        confianza_pred, clase_pred_idx = torch.max(probabilidades, dim=0)
        
    clase_predicha = clases[clase_pred_idx.item()]
    confianza_porcentaje = confianza_pred.item() * 100
    
    # 4. Mostrar la imagen en pantalla con Matplotlib
    plt.figure(figsize=(5, 5))
    plt.imshow(imagen_pil)
    color_titulo = "green" # Puedes ajustarlo si sabes la clase real de antemano
    plt.title(f"Modelo: {nombre_modelo}\nPredicción: {clase_predicha.upper()} ({confianza_porcentaje:.1f}%)", 
              fontsize=12, fontweight='bold', color='darkblue')
    plt.axis('off')
    plt.show()
    
    # 5. Imprimir resultados detallados en consola
    print(f"\n==================================================")
    print(f" 🩺 RESULTADO DE INFERENCIA ({nombre_modelo.upper()})")
    print(f"==================================================")
    print(f"📁 Archivo analizado: {os.path.basename(ruta_imagen)}")
    print(f"🏆 Predicción Final:   {clase_predicha.upper()} ({confianza_porcentaje:.2f}% de confianza)")
    print(f"--------------------------------------------------")
    print("📊 Desglose de probabilidades por clase:")
    for idx, clase in enumerate(clases):
        prob = probabilidades[idx].item() * 100
        print(f"   - {clase:<12}: {prob:5.2f}%")
    print(f"==================================================")
    
    return clase_predicha, probabilidades.cpu().numpy()

# --- EJEMPLO DE USO ---
ruta_prueba = "/content/drive/MyDrive/Data/Curso de Redes Neuronales Convolucionales/GBM.jpeg"

predecir_y_mostrar_imagen(
    ruta_imagen=ruta_prueba, 
    nombre_modelo='ResNet50', 
    trainer_dict=trainers, 
    clases=MIS_CLASES_BRAIN, 
    transformacion_val=val_test_transforms
)
# %%[markdown]
# ## Localización de imágenes erradas de gliomas

def analizar_errores_con_gradcam(modelo, dataloader, dataset, clases, num_ejemplos=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelo.to(device)
    modelo.eval()
    
    idx_glioma = clases.index('glioma')
    idx_meningioma = clases.index('meningioma')
    idx_notumor = clases.index('notumor')
    
    rutas_archivos = [item[0] for item in dataset.samples]
    
    print("🔍 1. Escaneando el dataset de prueba en busca de errores...")
    rutas_g_a_m = [] 
    rutas_g_a_nt = []
    
    with torch.no_grad():
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            outputs = modelo(images)
            _, preds = torch.max(outputs, 1)
            
            for j in range(len(labels)):
                idx_global = i * dataloader.batch_size + j
                real = labels[j].item()
                pred = preds[j].item()
                
                # Prevenir errores si el último batch es más pequeño
                if idx_global < len(rutas_archivos): 
                    ruta = rutas_archivos[idx_global]
                    if real == idx_glioma and pred == idx_meningioma:
                        rutas_g_a_m.append(ruta)
                    elif real == idx_glioma and pred == idx_notumor:
                        rutas_g_a_nt.append(ruta)

    print(f"⚠️ Encontrados: {len(rutas_g_a_m)} Glioma->Meningioma | {len(rutas_g_a_nt)} Glioma->NoTumor")
    print("🔥 2. Generando mapas de activación (Grad-CAM)...")

    # Configurar Grad-CAM para ResNet50 (apuntando a la última capa convolucional)
    target_layers = [modelo.backbone.layer4[-1]]
    cam = GradCAM(model=modelo, target_layers=target_layers)

    # Transformaciones estándar para pasar la imagen al modelo
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def visualizar_lista_errores(rutas, titulo_error):
        if not rutas: return
        
        fig, axes = plt.subplots(1, min(len(rutas), num_ejemplos), figsize=(15, 5))
        if num_ejemplos == 1: axes = [axes] # Manejo de un solo plot
        fig.suptitle(f"Análisis Grad-CAM: {titulo_error}", fontsize=16, weight='bold')

        for idx, ruta in enumerate(rutas[:num_ejemplos]):
            # 1. Cargar imagen original para el fondo (rgb, float [0,1])
            img_pil = Image.open(ruta).convert('RGB')
            img_redimensionada = img_pil.resize((224, 224))
            rgb_img = np.float32(img_redimensionada) / 255
            
            # 2. Preparar tensor para el modelo
            input_tensor = transform(img_pil).unsqueeze(0).to(device)
            
            # 3. Generar máscara Grad-CAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
            grayscale_cam = grayscale_cam[0, :]
            
            # 4. Superponer mapa de calor en la imagen original
            visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
            
            axes[idx].imshow(visualization)
            axes[idx].set_title(f"Error {idx+1}")
            axes[idx].axis('off')
            
        plt.tight_layout()
        plt.show()

    # Mostrar resultados
    visualizar_lista_errores(rutas_g_a_m, "Real: GLIOMA | Predicción: MENINGIOMA")
    visualizar_lista_errores(rutas_g_a_nt, "Real: GLIOMA | Predicción: NOTUMOR")


# --- EJECUCIÓN ---
CLASES_UNIFICADAS = ['glioma', 'meningioma', 'notumor', 'pituitary']
modelo_evaluar = modelo_residual # Reemplaza con la variable de tu ResNet50

analizar_errores_con_gradcam(
    modelo=modelo_evaluar, 
    dataloader=test_loader, 
    dataset=test_dataset, 
    clases=CLASES_UNIFICADAS,
    num_ejemplos=3 # Cambia esto si quieres ver más o menos imágenes
)

# %%
