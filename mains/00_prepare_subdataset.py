# import os
# import shutil
# import pandas as pd
# from utilities.config_parser import load_config
# from utilities.logger import get_logger

# def run_preparation():
#     cfg = load_config()
#     # Initialize logger with explicit file path
#     logger = get_logger("00_PREPARE", log_dir=cfg['paths']['logs_dir'])
    
#     logger.info("Starting dataset subset preparation...")
    
#     raw_in = cfg['paths']['original_raw_dir']
#     ann_out = cfg['paths']['annotations_dir']
#     vid_out = cfg['paths']['videos_dir']
    
#     try:
#         # Simulate moving files and creating CSVs based on your config targets
#         logger.info(f"Mapping targets from {raw_in}")
#         logger.info(f"Writing CSV annotations to {ann_out}")
#         logger.info(f"Copying video frames to {vid_out}")
        
#         # [Your existing dataframe/copy logic goes here]
        
#         logger.info("Dataset subset preparation completed successfully.")
        
#     except Exception as e:
#         logger.error(f"Failed during dataset preparation: {str(e)}", exc_info=True)
#         raise e

# if __name__ == "__main__":
#     run_preparation()