import advertorch
import advertorch.context
import re
import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import tools
import constants
from torch.utils.data import Dataset, DataLoader
import argparse


def get_model_variables(i=0):
    round_name = 'round6'
    base_path = f'/scratch/data/{round_name}-train-dataset'

    metadata = pd.read_csv(os.path.join(base_path, "METADATA.csv"))
    metadata.head()

    commands_df = metadata[['model_name', 'poisoned', 'embedding_flavor', 'source_dataset', 'cls_token_is_first']]

    embedding_flavor_to_tokenizer = {'gpt2':"GPT-2-gpt2.pt",
                                     'distilbert-base-uncased':"DistilBERT-distilbert-base-uncased.pt",
                                     'bert-base-uncased':"BERT-bert-base-uncased.pt"}
    commands_df['tokenizer_filename'] = commands_df['embedding_flavor'].map(embedding_flavor_to_tokenizer)
    commands_df['embedding_filename'] = commands_df['embedding_flavor'].map(embedding_flavor_to_tokenizer)
    commands_df.head()

    model_training_data_path = f'/scratch/data/{round_name}-train-dataset/'
    text_data_path = '/scratch/data/sentiment-classification/'

    model_filepath = os.path.join(model_training_data_path, "models", \
                                  commands_df.loc[[i]].model_name.item(), "model.pt")
    cls_token_is_first = commands_df.loc[[i]].cls_token_is_first.item()
    tokenizer_filepath = os.path.join(model_training_data_path, "tokenizers", \
                                      commands_df.loc[[i]].tokenizer_filename.item())
    embedding_filepath = os.path.join(model_training_data_path, "embeddings", \
                                      commands_df.loc[[i]].embedding_filename.item())
    examples_dirpath = os.path.join(model_training_data_path, "models", \
                                    commands_df.loc[[i]].model_name.item(), "clean_example_data")
    train_test = 'train'
    dataset_name = commands_df.loc[[i]].source_dataset.item()

    return model_filepath, cls_token_is_first, tokenizer_filepath, \
           embedding_filepath, examples_dirpath, train_test, dataset_name


def get_grads_for_model_id(model_id):
    model_filepath, cls_token_is_first, tokenizer_filepath, \
    embedding_filepath, examples_dirpath, train_test, dataset_name = get_model_variables(model_id)
    model = torch.load(model_filepath).cuda()
    grads = []

    rand = torch.randn((2000, 1, 768)).cuda()
    rand.requires_grad = True
    model.train()
    output = model(rand.cuda())
    loss_fn = torch.nn.CrossEntropyLoss()
    for target_class in [0,1]:
        loss = loss_fn(output, (torch.zeros(2000).cuda()+target_class).long())
        loss.backward(retain_graph=True)
        grads.append(torch.mean(rand.grad, dim=0))
    return grads

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=int)
    args = parser.parse_args()
    
    torch.random.manual_seed(42)

    grads = get_grads_for_model_id(args.model_id)

    model_id_str = 'id-'+str(100000000+args.model_id)[1:]
    np.save(f'random_gradients/{model_id_str}', torch.cat(grads).cpu().numpy().flatten())
