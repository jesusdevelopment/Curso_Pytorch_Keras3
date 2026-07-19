# %% [markdown]
# ## Bibliotecas
!pip install portalocker>=2.0.0 
!pip install torchtext --upgrade
import torchtext
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Dbpedia
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader