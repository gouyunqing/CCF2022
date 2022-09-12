from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
from process_data import read_data
import random

class Dataset(Dataset):
    def __init__(self):


