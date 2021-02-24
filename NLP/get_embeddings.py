import os
from os.path import join as join
import torch
import model_factories
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import jsonpickle
import json
import transformers
import itertools
import argparse

###### HELPER FUNCTIONS #####

# 1. Get tokenizers and embeddings
tokenizers_folder = "/scratch/data/round5-train-dataset/tokenizers/"
embedding_flavor_to_tokenizer = {'gpt2':
                                     torch.load(tokenizers_folder+
                                                  "GPT-2-gpt2.pt"),
                                 'distilbert-base-uncased':
                                     torch.load(tokenizers_folder+
                                                  "DistilBERT-distilbert-base-uncased.pt"),
                                 'bert-base-uncased':
                                     torch.load(tokenizers_folder+
                                                  "BERT-bert-base-uncased.pt")}

embeddings_folder = "/scratch/data/round5-train-dataset/embeddings/"
embedding_flavor_to_embedding = {'gpt2':
                                     torch.load(embeddings_folder+
                                                  "GPT-2-gpt2.pt"),
                                 'distilbert-base-uncased':
                                     torch.load(embeddings_folder+
                                                  "DistilBERT-distilbert-base-uncased.pt"),
                                 'bert-base-uncased':
                                     torch.load(embeddings_folder+
                                                  "BERT-bert-base-uncased.pt")}

# 2. Get cls embedding for a given text or list of text and an embedding flavor
def get_cls_embedding(text, flavor, multiple_gpu=True):
    tokenizer = embedding_flavor_to_tokenizer[flavor]
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]
    tokens = tokenizer(text, max_length=max_input_length - 2,
                       padding=True, truncation=True, return_tensors="pt",
                       is_split_into_words=False)
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            embedding = embedding_flavor_to_embedding[flavor].to(device)
            if multiple_gpu:
                embedding = torch.nn.DataParallel(embedding)
            embedding_vector = embedding(tokens.input_ids.to(device),
                                         attention_mask=tokens.attention_mask.to(device))[0]
    if 'bert' in flavor:
        return(embedding_vector[:,0,:].cpu().numpy())
    else:
        eos = np.argmin(tokens.attention_mask.numpy(), axis=1)
        eos[eos == 0] = -1
        return(embedding_vector[np.arange(len(embedding_vector)),eos,:].cpu().numpy())

#### MAIN ####

# 0. iterate through each dataset, flavor and train/test
dataset_base_path = '/scratch/data/sentiment-classification'
dataset_path_list = [join(dataset_base_path, f) for f in os.listdir(dataset_base_path) 
                                                    if os.path.isdir(join(dataset_base_path, f))]
embedding_flavor_list = list(embedding_flavor_to_embedding.keys()) 

combinations =  itertools.product(dataset_path_list, embedding_flavor_list, ['train', 'test'])    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, 
                        help='Batch size to process embeddings', 
                        default=32)
    parser.add_argument('--dataset_path', type=str, 
                        help='Where is the sentiment classification datasets stored',
                        default='/scratch/data/sentiment-classification')
    parser.add_argument('--embedding_flavor', type=str, 
                        help='What embedding flavor to use', 
                        choices=list(embedding_flavor_to_embedding.keys()))
    parser.add_argument('--train_test', type=str, 
                        help='Specify if we are getting embeddings for train or test sets', 
                        choices=['train', 'test'])
    args = parser.parse_args()
    
    # 1. read json
    df = pd.DataFrame([])
    with open(join(args.dataset_path, args.train_test+'.json')) as f:
        json_df = json.load(f)
        df = pd.DataFrame.from_dict(json_df, orient='index').reset_index()

    # 2. make folder for the npy files
    embedding_folder_path = join(args.dataset_path, args.embedding_flavor)
    train_test_folder_path = join(embedding_folder_path, args.train_test)
    if args.embedding_flavor not in os.listdir(args.dataset_path):
        os.mkdir(embedding_folder_path)
    if args.train_test not in os.listdir(embedding_folder_path):
        os.mkdir(train_test_folder_path)
    
    # 3. calculate embeddings and save them
    old_i = 0
    for i in range(args.batch_size, len(df), args.batch_size):
        text = list(df.loc[old_i:i-1]['data'])
        embedding = get_cls_embedding(text, args.embedding_flavor)
        np.save(join(train_test_folder_path, f'{old_i}_to_{i}'), embedding)
        old_i = i

    # 4. save the last batch with a single gpu
    for i in range(old_i, len(df)+32, 32):
        text = list(df.loc[old_i:i-1]['data'])
        if len(text) < 1: 
            break
        embedding = get_cls_embedding(text, args.embedding_flavor, multiple_gpu=False)
        np.save(join(train_test_folder_path, f'{old_i}_to_{i}'), embedding)
        old_i = i


