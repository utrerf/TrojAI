import os
from os.path import join as join
import pandas as pd
import tools
import shutil


NUM_CLEAN_TRAINING_MODELS = 3
# All other models are used for testing

''' 1. Load metadata '''
TRAINING_DATA_PATH = tools.TRAINING_DATA_PATH

metadata = pd.read_csv(join(TRAINING_DATA_PATH, 'METADATA.csv'))
clean_metadata = metadata[metadata['poisoned'] == False]

''' 2. Get the models with the highest clean test acc per embedding and dataset combination '''
# groupby = clean_metadata\
#           .groupby(['embedding_flavor', 'source_dataset'], as_index=False)\
#           .agg({'final_clean_data_test_acc': 'max'})
train_df = clean_metadata\
          .groupby(['embedding_flavor', 'source_dataset'], as_index=False)\
          .apply(lambda x: x.nlargest(NUM_CLEAN_TRAINING_MODELS-1, 'final_clean_data_test_acc'))\
          .reset_index(drop=True)
train_df2 = clean_metadata\
          .groupby(['embedding_flavor', 'source_dataset'], as_index=False)\
          .apply(lambda x: x.nsmallest(1, 'final_clean_data_test_acc'))\
          .reset_index(drop=True)
train_df = train_df.append(train_df2).reset_index(drop=True)    

test_df = clean_metadata\
          .groupby(['embedding_flavor', 'source_dataset'], as_index=False)\
          .apply(lambda x: x.nlargest(100, 'final_clean_data_test_acc'))\
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
    models_folder = join(TRAINING_DATA_PATH, 'models')

    for i in range(len(df)):
        entry = df.loc[i]
        model_id = entry['model_name']
        source_path = join(models_folder, model_id)
        config = tools.load_config(join(source_path, 'model.pt'))
        dataset = config['source_dataset'].lower()
        embedding = config['embedding']
        
        destination = join(dest_foldername, f'{dataset}_{embedding}_{model_id}')
        shutil.copytree(source_path, destination)


copy_models('clean_models_train', train_df)
copy_models('clean_models_test', test_df)


print('end')