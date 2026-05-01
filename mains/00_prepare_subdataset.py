# import os
# import shutil
# import pandas as pd
# from utilities.config_parser import load_config
# from utilities.logger import get_logger
# from utilities.reproducibility import lock_seeds

# logger = get_logger("00_SUBSET")

# def run_subsetting():
#     cfg = load_config()
#     lock_seeds(cfg['experiment']['seed'])
    
#     raw_in = cfg['paths']['original_raw_dir']
#     raw_out = cfg['paths']['raw_data_dir']
#     split_out = cfg['paths']['splits_dir']
#     os.makedirs(raw_out, exist_ok=True)
#     os.makedirs(split_out, exist_ok=True)

#     # Simplified representation: This copies folders from original_raw_dir to raw_out
#     # and generates the train/val/test CSVs in split_out
#     logger(f"Subsampling raw data to {raw_out} and creating CSVs in {split_out}")

# if __name__ == "__main__":
#     run_subsetting()