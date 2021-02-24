import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from robustness.datasets import DataSet 


def read_examples_into_list(examples_dirpath):
    text_list = []
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) 
                                                              if fn.endswith('.txt')]
    for fn in fns:
        with open(fn, 'r') as fh:
            text_list.append(fh.readline())
    return text_list


def determine_dataset_name(text_list, char_df):
    dataset_counter = {}
    for text in text_list:
        #possible_datasets = list(char_df[char_df['data'].str.contains(text[:32], na=False)]['dataset'])
        possible_datasets = set(char_df[char_df['data'] == text[:32]]['dataset'])
        for dataset in possible_datasets:
            if dataset in dataset_counter.keys():
                dataset_counter[dataset] += 1
            else:
                dataset_counter[dataset] = 1
    df = pd.DataFrame(dataset_counter.items(), columns = ['dataset', 'count'])
    return df.loc[[df['count'].argmax()]]['dataset'].item()


def determine_embedding_flavor(embedding):
    if 'Distil' in embedding._get_name():
        return 'distilbert-base-uncased'
    elif 'GPT' in embedding._get_name():
        return 'gpt2'
    else:
        return 'bert-base-uncased'

class TrojAIDataset(Dataset):
    """ Class used to make the dataset from examples_dirpath """
    
    def __init__(self, dataset_name, embedding_flavor, embeddings_base_path, train_test='train'):
        # 1. load train or test embeddings
        embedding_path = os.path.join(embeddings_base_path, dataset_name, 
                                      embedding_flavor, train_test, 'concatenated_embedding.npy')
        self.x = torch.from_numpy(np.load(embedding_path))

        # 2. load the corresponding labels
        labels_path = os.path.join(embeddings_base_path, dataset_name, 
                                               f'{train_test}_labels.npy')
        self.y_np = np.load(labels_path)
        self.y = torch.from_numpy(self.y_np)
        self.y = torch.LongTensor(self.y)

        # 3. constrain to min dimension
        min_dim = min(self.y.shape[0], self.x.shape[0])
        self.x = self.x[:min_dim, :]
        self.y = self.y[:min_dim]
        self.y_np = self.y_np[:min_dim]
        print(f'y shape: {self.y.shape}')
        print(f'y sum: {sum(self.y)}')
    
    def __getitem__(self, index):
        x, y = torch.unsqueeze(self.x[index], 0), self.y[index]
        #TODO: Implement transform
        #if self.transform: 
        #    x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.x)
    
    def positive_sentiment_ix(self):
        return np.argwhere(self.y_np)
    
    def negative_sentiment_ix(self):
        return np.argwhere(np.where(a == 0, 1, 0))

def make_weighted_sampler(ds):
    class_sample_count = np.array(
                         [len(np.where(ds.y == t)[0]) for t in np.unique(ds.y)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in ds.y])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weigth = samples_weight.double()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler



