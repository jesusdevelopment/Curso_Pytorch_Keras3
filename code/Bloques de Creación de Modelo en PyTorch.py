# %% [markdown]
# ## Importación de Bibliotecas

import torch
from torch import nn
import matplotlib.pyplot as plt

# Comprobar la versión de PyTorch
torch.__version__

# %% [markdown]
## Creación de un Tensor Simple
# Crea *nuevos* parámetros
volumen = 0.8
sesgo = 0.2

# Crea datos
inicio = 0
final = 1
step = 0.025
X = torch.arange(inicio, final, step).unsqueeze(dim=1)
print(X)
print(f"Shape de X: {X.shape}")
y = volumen * X + sesgo
print(f"Shape de y: {y.shape}")

X[0:10], y[0:10]
# %% [markdown]
## División de Datos en Entrenamiento y Prueba
# Es crucial dividir los datos para evaluar el rendimiento del modelo
# Crea la división

train_division = int(0.7 * len(X)) # 70% de los datos utilizados para el conjunto de entrenamiento, 30% para pruebas
X_ent, y_ent = X[:train_division], y[:train_division]
X_prueb, y_prueb = X[train_division:], y[train_division:]

len(X_ent), len(X_prueb)
# %%

def plot_predictions(datos_ent=X_ent, 
                     etiq_ent=y_ent, 
                     datos_prueba=X_prueb, 
                     etiq_prueba=y_prueb, 
                     predictions=None):
    """
    Traza datos de entrenamiento, datos de prueba y compara predicciones.
    Soporta tensores de PyTorch convirtiéndolos automáticamente a NumPy.
    """
    # Conversión segura de PyTorch a NumPy para evitar errores de renderizado
    if isinstance(datos_ent, torch.Tensor): datos_ent = datos_ent.detach().cpu().numpy()
    if isinstance(etiq_ent, torch.Tensor): etiq_ent = etiq_ent.detach().cpu().numpy()
    if isinstance(datos_prueba, torch.Tensor): datos_prueba = datos_prueba.detach().cpu().numpy()
    if isinstance(etiq_prueba, torch.Tensor): etiq_prueba = etiq_prueba.detach().cpu().numpy()
    
    plt.figure(figsize=(10, 10))

    # Traza datos de entrenamiento en verde
    plt.scatter(datos_ent, etiq_ent, c="g", s=6, label="Datos de entrenamiento")
  
    # Traza datos de prueba en amarillo/naranja
    plt.scatter(datos_prueba, etiq_prueba, c="orange", s=10, label="Datos de prueba")

    if predictions is not None:
        # Si hay predicciones, las convierte a numpy y las grafica en rojo
        if isinstance(predictions, torch.Tensor): 
            predictions = predictions.detach().cpu().numpy()
        plt.scatter(datos_prueba, predictions, c="r", s=10, label="Predicciones")

    # Configuración de la leyenda y visualización
    plt.legend(prop={"size": 12})
    
    # NUEVO: Forzar a Matplotlib a renderizar y mostrar la ventana con la gráfica
    plt.show()

# %% 2. Llamada a la función (¡Ojo a la indentación!)
# Debe estar completamente al margen izquierdo, sin espacios iniciales.
plot_predictions()
# %% [markdown]
# Creación de la clase 
class ModeloRegresionLineal(nn.Module):
    def __init__(self):
        super().__init__()
        self.volumen=nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias=nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
    def forward(self, x:torch.Tensor)->torch.Tensor:
        return self.volumen * x + self.bias