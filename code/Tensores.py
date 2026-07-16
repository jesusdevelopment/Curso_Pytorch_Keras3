# %% [markdown]
import torch
# 1. Creación de Tensores básicos
# Crear un tensor desde una lista
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)
print(x_data)
print(x_data.shape)
print(x_data.dtype)
print(x_data.device)
print(x_data.size)
print(torch.__version__)

# %%
# 2. Creación de tensores con formas específicas    
escalar = torch.rand(1)
print(escalar)
vector = torch.zeros(1, 10)
print(vector)
print(vector.shape)
matriz = torch.ones(2, 2)
print(matriz)
print(matriz.shape)
# %%

t5 = torch.rand(5, 2, 3)
print(t5)



# %%
t0 = torch.tensor(200) 
print(t0)
print(t0.shape)
t1 = torch.tensor([1, 2, 3]) 
print(t1)
print(t1.shape)
t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(t2)
print(t2.shape)
# %%
matriz_float32 = torch.tensor([[3.1,3.2], [3.3,3.4]])
matriz_uint64 = torch.tensor([[3,3], [3,3]])

(matriz_float32 + matriz_uint64).dtype
# %%
if torch.cuda.is_available():
  matriz_uint64_cuda = matriz_uint64.to(torch.device("cuda"))

  print(matriz_uint64_cuda, matriz_uint64_cuda.type())
  print(matriz_uint64_cuda.to("cpu", torch.float32))

  #matriz_uint64_cuda + matriz_uint64
# %%
matriz.numpy()
type(matriz.numpy())
# %%

# create a tensor of zeros with shape (3, 4)
zeros_tensor = torch.zeros((3, 4))

# create a tensor of ones with shape (3, 4)
ones_tensor = torch.ones((3, 4))

# create a tensor of random values with shape (4,)
random_tensor = torch.randn((4))

print(random_tensor)
# %%

# add two tensors element-wise
added_tensor = zeros_tensor + ones_tensor
print(added_tensor)
# subtract two tensors element-wise
subtracted_tensor = zeros_tensor - ones_tensor
print(subtracted_tensor)
# multiply two tensors element-wise
multiplied_tensor = zeros_tensor * ones_tensor
print(multiplied_tensor)
# divide two tensors element-wise
divided_tensor = random_tensor / ones_tensor
print(divided_tensor)
# %%
# create two matrices
matrix1 = torch.randn(2,3)
matrix2 = torch.randn(3,2)

print(f"matrix1 shape: {matrix1.shape}")
print(f"matrix2 shape: {matrix2.shape}")

# perform matrix multiplication
print(torch.matmul(matrix1, matrix2).shape)
matrix1@matrix2
matrix1*matrix2
# %%
