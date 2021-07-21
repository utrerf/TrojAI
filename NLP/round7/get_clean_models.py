import os
from os.path import join as join
import pandas as pd
import tools
import shutil

NUM_CLEAN_MODELS = 8

''' 1. Load metadata '''
TRAINING_DATA_PATH = tools.TRAINING_DATA_PATH

metadata = pd.read_csv(join(TRAINING_DATA_PATH, 'METADATA.csv'))
clean_metadata = metadata[metadata['poisoned'] == False]

''' 2. Get the models with the highest clean test acc per embedding and dataset combination '''
# groupby = clean_metadata\
#           .groupby(['embedding_flavor', 'source_dataset'], as_index=False)\
#           .agg({'final_clean_data_test_acc': 'max'})
groupby = clean_metadata\
          .groupby(['embedding_flavor', 'source_dataset'], as_index=False)\
          .apply(lambda x: x.nlargest(NUM_CLEAN_MODELS, 'final_clean_data_test_acc'))\
          .reset_index(drop=True)

# result = groupby.merge(metadata, how='left', 
#                        on=['embedding_flavor', 'source_dataset', 'final_clean_data_test_acc'])

''' 3. Move those models to a new folder '''
def check_if_folder_exists(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)

check_if_folder_exists('clean_models')

models_folder = join(TRAINING_DATA_PATH, 'models')

for i in range(len(groupby)):
    entry = groupby.loc[i]
    model_id = entry['model_name']
    source_path = join(models_folder, model_id)
    config = tools.load_config(join(source_path, 'model.pt'))
    dataset = config['source_dataset'].lower()
    embedding = config['embedding']
    
    destination = join('clean_models', f'{dataset}_{embedding}_{model_id}')
    shutil.copytree(source_path, destination)
    

print('end')