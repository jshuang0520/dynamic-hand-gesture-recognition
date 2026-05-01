import os
import random
import numpy as np
import torch
from utilities.logger import get_logger

log = get_logger("REPRODUCIBILITY")

def lock_seeds(seed=20260430):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    log(f"Universal random seed locked to {seed}")