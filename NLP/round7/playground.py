import argparse
import torch
import json
import os
from os.path import join as join
import model_factories
import numpy as np
import pandas as pd


def tokenize_and_align_labels(tokenizer, original_words, 
                              original_labels, max_input_length):

    tokenized_inputs = tokenizer(original_words, padding=True, truncation=True, 
                                 is_split_into_words=True, max_length=max_input_length)
    
    labels, label_mask = [], []
    word_ids = [tokenized_inputs.word_ids(i) for i in range(len(original_labels))]
    previous_word_idx = None

    for i, sentence in enumerate(word_ids):
        temp_labels, temp_mask = [], []
        for word_idx in sentence:
            if word_idx is not None:
                cur_label = original_labels[i][word_idx]
            if word_idx is None:
                temp_labels.append(-100)
                temp_mask.append(0)
            elif word_idx != previous_word_idx:
                temp_labels.append(cur_label)
                temp_mask.append(1)
            else:
                temp_labels.append(-100)
                temp_mask.append(0)
            previous_word_idx = word_idx
        labels.append(temp_labels)
        label_mask.append(temp_mask)
        
    return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], \
           labels, label_mask


def load_config(model_filepath):

    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
    print('Source dataset name = "{}"'.format(config['source_dataset']))
    if 'data_filepath' in config.keys():
        print('Source dataset filepath = "{}"'.format(config['data_filepath']))
    return config


def get_max_input_length(config, tokenizer):
    if config['embedding'] == 'MobileBERT':
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    else:
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]
    return max_input_length


def get_words_and_labels(examples_dirpath):
    fns = [os.path.join(examples_dirpath, fn) \
           for fn in os.listdir(examples_dirpath) \
           if fn.endswith('.txt')]
    fns.sort()
    original_words = []
    original_labels = []

    for fn in fns:
        if fn.endswith('_tokenized.txt'):
            continue
        # load the example
        with open(fn, 'r') as fh:
            lines = fh.readlines()
            temp_words = []
            temp_labels = []

            for line in lines:
                split_line = line.split('\t')
                word = split_line[0].strip()
                label = split_line[2].strip()
                
                temp_words.append(word)
                temp_labels.append(int(label))
        original_words.append(temp_words)
        original_labels.append(temp_labels)

    return original_words, original_labels


def to_tensor_and_device(var, device):
    var = torch.as_tensor(var)
    var = var.to(device)
    # var = torch.unsqueeze(var, axis=0)
    return var


def predict_sentiment(classification_model, input_ids, 
                      attention_mask, labels, labels_mask):

    _, logits = classification_model(input_ids, 
                                     attention_mask=attention_mask, 
                                     labels=labels)        
    preds = torch.argmax(logits, dim=2).squeeze()
    
    masked_labels = labels.view(-1)[labels_mask.view(-1)]
    masked_preds = preds.view(-1)[labels_mask.view(-1)]
    
    n_correct = torch.eq(masked_labels, masked_preds).sum()
    n_total = labels_mask.sum()

    return preds, n_correct, n_total


def show_predictions_vs_originals(predicted_labels, original_words):
    for i, sentence in enumerate(original_words):
        print(f'Original Words: {sentence}')
        print(f'Predicted Labels: {predicted_labels[i]}')


def trojan_detector(model_filepath, tokenizer_filepath, 
                    result_filepath, scratch_dirpath, examples_dirpath):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = False 

    config = load_config(model_filepath)
    classification_model = torch.load(model_filepath, 
                                      map_location=torch.device(device))
    tokenizer = torch.load(tokenizer_filepath)
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    max_input_length = get_max_input_length(config, tokenizer)

    original_words, original_labels = get_words_and_labels(examples_dirpath)

    input_ids, attention_mask, \
        labels, labels_mask = tokenize_and_align_labels(tokenizer, original_words, 
                                                        original_labels, max_input_length)

    input_ids = to_tensor_and_device(input_ids, device)
    attention_mask = to_tensor_and_device(attention_mask, device)
    labels = to_tensor_and_device(labels, device)
    labels_mask = to_tensor_and_device(labels_mask, device)

    predicted_labels, n_correct, \
        n_total = predict_sentiment(classification_model, input_ids, 
                                    attention_mask, labels, labels_mask)

    show_predictions_vs_originals(predicted_labels, original_words)
    print(f'Correct: {n_correct} Total: {n_total} Accuracy: {n_correct/n_total}')
    assert len(predicted_labels) == len(original_words)
    
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trojan Detector for Round 7.')

    #TODO: REMEMBER TO CHANGE DEFAULT OF IS_TRAINING BACK TO 0
    parser.add_argument('--is_training', type=int, choices=[0, 1], 
                        help='Helps determine if we are training or testing.'\
                             ' If training just specify model number', 
                        default=1)
    parser.add_argument('--model_num', type=int, 
                        help='Model id number', 
                        default=0)
    parser.add_argument('--training_data_path', type=str, 
                        help='Folder that contains the training data', 
                        default='/scratch/data/TrojAI/round7-train-dataset/')
    parser.add_argument('--model_filepath', type=str, 
                        help='File path to the pytorch model file to be evaluated.', 
                        default='/scratch/data/TrojAI/round7-train-dataset/models/id-00000000')
    parser.add_argument('--tokenizer_filepath', type=str, 
                        help='File path to the pytorch model (.pt) file containing the '\
                             'correct tokenizer to be used with the model_filepath.', 
                        default='./model/tokenizer.pt')
    parser.add_argument('--result_filepath', type=str, 
                        help='File path to the file where output result should be written. '\
                             'After execution this file should contain a single line with a'\
                             ' single floating point trojan probability.', 
                        default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, 
                        help='File path to the folder where scratch disk space exists. '\
                             'This folder will be empty at execution start and will be '\
                             'deleted at completion of execution.', 
                        default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, 
                        help='File path to the folder of examples which might be useful '\
                             'for determining whether a model is poisoned.', 
                        default='./test-model/clean_example_data')

    args = parser.parse_args()

    if args.is_training:
        metadata = pd.read_csv(join(args.training_data_path, 'METADATA.csv'))

        id_str = str(100000000 + args.model_num)[1:]
        model_id = 'id-'+id_str

        data = metadata[metadata.model_name==model_id]
        
        # get the tokenizer name
        embedding_level = data.embedding.item()
        tokenizer_name = data.embedding_flavor.item().replace("/", "-")
        full_tokenizer_name = embedding_level+'-'+tokenizer_name+'.pt'

        args.model_filepath = join(args.training_data_path, 'models', model_id, 'model.pt')
        args.tokenizer_filepath = join(args.training_data_path, 'tokenizers', full_tokenizer_name)
        args.examples_dirpath = join(args.training_data_path, 'models', model_id, 'clean_example_data')

    trojan_detector(args.model_filepath, 
                    args.tokenizer_filepath, 
                    args.result_filepath, 
                    args.scratch_dirpath,
                    args.examples_dirpath)

