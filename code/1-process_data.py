import pandas as pd
from utils import processing_functions
import yaml


with open('../code/astros_config.yaml', 'r') as file:
    config = yaml.safe_load(file)


df = processing_functions.download_data(config)
processing_functions.process_data(pitch_ids=df['pitch_ids'].values,keep_wavs=False)