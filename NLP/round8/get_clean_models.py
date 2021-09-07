import os
from os.path import join as join
import pandas as pd
import shutil
import detector


NUM_CLEAN_TRAINING_MODELS = 1
# All other models are used for testing

''' 1. Load metadata '''
metadata = pd.read_csv(join(detector.TRAINING_FILEPATH, 'METADATA.csv'))
clean_metadata = metadata[metadata['poisoned'] == False]

''' 2. Get the models with the highest clean test acc per embedding and dataset combination '''
train_df = clean_metadata\
          .groupby(['model_architecture', 'source_dataset'], as_index=False)\
          .apply(lambda x: x.nsmallest(NUM_CLEAN_TRAINING_MODELS, 'test_clean_loss'))\
          .reset_index(drop=True)

test_df = clean_metadata\
          .groupby(['model_architecture', 'source_dataset'], as_index=False)\
          .apply(lambda x: x.nsmallest(100, 'test_clean_loss'))\
          .reset_index(drop=True)
test_df = test_df[~test_df['model_name'].isin(train_df['model_name'])].reset_index(drop=True)

# result = groupby.merge(metadata, how='left', 
#                        on=['embedding_flavor', 'source_dataset', 'final_clean_data_test_acc'])

''' 3. Move those models to a new folder '''
def check_if_folder_exists(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)


def copy_models(dest_foldername, df):
    check_if_folder_exists(dest_foldername)
    models_folder = join(detector.TRAINING_FILEPATH, 'models')

    for i in range(len(df)):
        entry = df.loc[i]
        model_id = entry['model_name']
        source_path = join(models_folder, model_id)
        config = detector.load_config(join(source_path, 'model.pt'))
        dataset = config['source_dataset'].lower()
        model_architecture = config['model_architecture']
        model_architecture = model_architecture.split('/')[-1]
        
        destination = join(dest_foldername, f'{dataset}_{model_architecture}_{model_id}')
        shutil.copytree(source_path, destination)


copy_models('clean_models_train', train_df)
copy_models('clean_models_test', test_df)


print('end')