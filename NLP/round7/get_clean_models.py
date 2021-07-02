import os
from os.path import join as join
import pandas as pd
import tools
import shutil

TRAINING_DATA_PATH = tools.TRAINING_DATA_PATH

metadata = pd.read_csv(join(TRAINING_DATA_PATH, 'METADATA.csv'))
clean_metadata = metadata[metadata['poisoned'] == False]

groupby = clean_metadata\
          .groupby(['embedding_flavor', 'source_dataset'], as_index=False)\
          .agg({'final_clean_data_test_acc': 'max'})

result = groupby.merge(metadata, how='left', 
                       on=['embedding_flavor', 'source_dataset', 'final_clean_data_test_acc'])


def check_if_folder_exists(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)

check_if_folder_exists('clean_models')

model_folder = join(TRAINING_DATA_PATH, 'models')

for i in range(len(result)):
    entry = result.loc[i]
    embedding = entry['embedding_flavor']
    dataset = entry['source_dataset']
    model_id = entry['model_name']
    source = join(model_folder, model_id)
    destination = join('clean_models', f'{dataset}_{embedding}_{model_id}')
    shutil.copytree(source, destination)
    

print('end')