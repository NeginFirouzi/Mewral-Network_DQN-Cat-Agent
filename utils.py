# utils.py
import os
import random
import numpy as np
import torch
from config import SEED

def set_seed(seed: int = None) -> None:
    """
    Initialize random seeds for reproducibility.
    """
    seed = SEED if seed is None else seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # enforce deterministic behavior in cuDNN (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark   = False
    torch.use_deterministic_algorithms(True)