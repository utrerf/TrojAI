import pandas as pd
import numpy as np
import os
import shutil

TRAINING_DATA_PATH = '/scratch/data/TrojAI/round7-train-dataset/'

if __name__ == "__main__":

    metadata = pd.read_csv(os.path.join(TRAINING_DATA_PATH, 'METADATA.csv'))

    datasets_path = os.path.join(TRAINING_DATA_PATH, 'datasets')

    # make all datasets folders
    if 'datasets' not in os.listdir(TRAINING_DATA_PATH):    
        os.mkdir(datasets_path)

    for dataset in metadata.source_dataset.unique():
        if dataset not in os.listdir(datasets_path):
            os.mkdir(os.path.join(datasets_path, dataset))
    
    # copy all clean example files to their respective dataset
    for id in metadata.model_name.unique():
        model_data_path = os.path.join(TRAINING_DATA_PATH, 'models', id, 'clean_example_data')
        data_files = [f for f in os.listdir(model_data_path) if 'tokenized' not in f]
        data_filepaths = [os.path.join(model_data_path, f) for f in data_files]

        for data_filepath, data_filename in zip(data_filepaths, data_files):
            new_data_filename = f'{data_filename}_{id}'
            dataset_name = metadata[metadata['model_name'] == id]['source_dataset'].item()
            shutil.copy(data_filepath, os.path.join(datasets_path, dataset_name, new_data_filename))
    