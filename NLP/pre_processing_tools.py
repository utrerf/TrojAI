import os
import pandas as pd
import json
import re
import numpy as np
import constants

def get_32_char_df(base_path=constants.datasets_base_path):
    df_list = []
    
    dataset_list = [x for x in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, x))]
    for dataset in dataset_list:
        for train_test in ['train', 'test']:
            file_path = os.path.join(base_path, dataset, f'{train_test}.json')
            with open(file_path) as f:
                js = json.load(f)
                df = pd.DataFrame.from_dict(js, orient='index').reset_index()[['data']]
                df['data'] = df['data'].str.slice(0,32)
                df['dataset'] = dataset
                df_list.append(df)

    return pd.concat(df_list)

def sort_func(e):
    return int(re.findall(r'(\d+)_to_', e)[0])

def aggregate_embeddings(dataset_path=constants.datasets_base_path):
    flavor_path_list = [os.path.join(dataset_path, x) for x in os.listdir(dataset_path) 
                                      if os.path.isdir(os.path.join(dataset_path, x))]
    for flavor_path in flavor_path_list:
        for train_test in ['train', 'test']:
            train_test_path = os.path.join(flavor_path, train_test)
            npy_file_list = [os.path.join(train_test_path, x) for x in os.listdir(train_test_path) 
                                                                                   if '_to_' in x]
            npy_file_list.sort(key=sort_func)
            embedding_list = []
            for f in npy_file_list:
                embedding_list.append(np.load(f))
            concatenated_embedding = np.concatenate(embedding_list)
            np.save(os.path.join(train_test_path, 'concatenated_embedding'), concatenated_embedding)

def split_out_labels(base_path=constansts.datasets_base_path):
    dataset_path_list = [os.path.join(base_path, x) for x in os.listdir(base_path) 
                                      if os.path.isdir(os.path.join(base_path, x))]
    for dataset_path in dataset_path_list: 
        for train_test in ['train', 'test']:
            file_path = os.path.join(dataset_path, f'{train_test}.json')
            with open(file_path) as f:
                js = json.load(f)
                df = pd.DataFrame.from_dict(js, orient='index').reset_index()[['label']]
                np.save(os.path.join(dataset_path, f'{train_test}_labels'), np.array(list(df['label'])))


if __name__ == '__main__':
    get_32_char_df()
    split_out_labels()

    base_path = '/scratch/data/sentiment-classification'
    dataset_path_list = [os.path.join(constants.datasets_base_path, x) for x in os.listdir(constants.datasets_base_path) 
                                                         if os.path.isdir(os.path.join(constants.datasets_base_path, x))]
    for ds in dataset_path_list:
        aggregate_embeddings(ds)


