import logging
import os
from datetime import datetime

def get_logger(name, log_dir=None):
    """
    Creates a logger that writes to both the console and a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if the logger already exists
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s', 
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 1. Console Handler (for SLURM .out files)
        c_handler = logging.StreamHandler()
        c_handler.setFormatter(formatter)
        logger.addHandler(c_handler)

        # 2. File Handler (for dedicated Python .log files)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
            f_handler = logging.FileHandler(log_file)
            f_handler.setFormatter(formatter)
            logger.addHandler(f_handler)

    return logger