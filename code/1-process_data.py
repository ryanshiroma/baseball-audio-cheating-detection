import pandas as pd
from utils import processing_functions
import yaml
import os
PITCH_TABLE_PATH = '../data'

with open('../code/astros_config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# df = processing_functions.download_data(config)
# df = pd.read_csv(os.path.join(PITCH_TABLE_PATH,'pitch_table.csv'))
processing_functions.process_data(keep_wavs=False)