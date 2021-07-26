import json
import os
import torch
from os.path import join as join
import pandas as pd
from copy import deepcopy
import transformers
import re
import numpy as np
import random


''' CONSTANTS '''
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EXTRACTED_GRADS = []
EXTRACTED_CLEAN_GRADS = []
TRAINING_DATA_PATH = '/scratch/data/TrojAI/round7-train-dataset/'
CLEAN_MODELS_PATH = '/scratch/utrerf/TrojAI/NLP/round7/clean_models_train'
TESTING_CLEAN_MODELS_PATH = '/scratch/utrerf/TrojAI/NLP/round7/clean_models_test'
LOGITS_CLASS_MASK = None
LOGITS_CLASS_MASK_CLEAN = None
# TODO: Change the number of clean models at eval to something that makes more sense
NUM_CLEAN_MODELS_AT_EVAL = 8
IS_TARGETTED = True
SIGN = 1 # min loss if we're targetting a class
if IS_TARGETTED:
    SIGN = -1 
LAMBDA = .75
USE_AMP = True

def get_logit_class_mask(class_list, classification_model, add_zero=False):
    if add_zero:
        class_list = [0] + class_list
    logits_class_mask = torch.zeros([classification_model.num_labels, len(class_list)])
    for new_cls, old_cls in enumerate(class_list):
        logits_class_mask[old_cls][new_cls] = 1
        if not add_zero or new_cls!=0:
            logits_class_mask[old_cls+1][new_cls] = 1
    return logits_class_mask


def modify_args_for_training(args):
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
    return args


def get_class_list(examples_dirpath):
    file_list = os.listdir(examples_dirpath)
    class_set = set()
    for f in file_list:
        class_num = int(re.findall(r'class_(\d+)_', f)[0])
        class_set.add(class_num)
    return list(class_set)


def load_config(model_filepath):
    model_filepath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_filepath, 'config.json')) as json_file:
        config = json.load(json_file)
    print('Source dataset name = "{}"'.format(config['source_dataset']))
    if 'data_filepath' in config.keys():
        print('Source dataset filepath = "{}"'.format(config['data_filepath']))
    return config


def load_tokenizer(tokenizer_filepath, config):
    if config['embedding'] == 'RoBERTa':
        tokenizer = \
            transformers.AutoTokenizer.from_pretrained(config['embedding_flavor'], 
                                                       use_fast=True, add_prefix_space=True)
    else:
        tokenizer = torch.load(tokenizer_filepath)
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_max_input_length(config, tokenizer):
    if config['embedding'] == 'MobileBERT':
        max_input_length = \
            tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    else:
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]
    return max_input_length


def get_embedding_weight(classification_model):
    word_embedding = find_word_embedding_module(classification_model)
    return deepcopy(word_embedding.weight)


def find_word_embedding_module(classification_model):
    word_embedding_tuple = [(name, module) 
        for name, module in classification_model.named_modules() 
        if 'embeddings.word_embeddings' in name]
    assert len(word_embedding_tuple) == 1
    return word_embedding_tuple[0][1]


def add_hooks(classification_model, is_clean):
    module = find_word_embedding_module(classification_model)
    module.weight.requires_grad = True
    if is_clean:
        module.register_backward_hook(extract_clean_grad_hook)
    else:
        module.register_backward_hook(extract_grad_hook)


def extract_grad_hook(module, grad_in, grad_out):
    EXTRACTED_GRADS.append(grad_out[0])  


def extract_clean_grad_hook(module, grad_in, grad_out):
    EXTRACTED_CLEAN_GRADS.append(grad_out[0])  


def get_words_and_labels(examples_dirpath, source_class=None):
    fns = [os.path.join(examples_dirpath, fn) \
           for fn in os.listdir(examples_dirpath) \
           if fn.endswith('.txt')]
    if source_class is not None:
        fns = [i for i in fns if f'class_{source_class}' in i]
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


def tokenize_and_align_labels(tokenizer, original_words, 
                              original_labels, max_input_length):

    tokenized_inputs = tokenizer(original_words, padding=True, truncation=True, 
                                 is_split_into_words=True, max_length=max_input_length)
    
    word_ids = [tokenized_inputs.word_ids(i) for i in range(len(original_labels))]
    labels, label_mask = [], []
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


def eval_batch_helper(clean_models, classification_model, all_vars, source_class_token_locations,
                      source_class=0, target_class=0, clean_class_list=[], class_list=[], 
                      num_triggers_in_batch=1, is_triggered=True):
    clean_logits_list = []
    # for clean_model in random.sample(clean_models, min(len(clean_models), NUM_CLEAN_MODELS_AT_EVAL)):
    for clean_model in clean_models:
        clean_logits_list.append(clean_model(all_vars['input_ids'], all_vars['attention_mask'], num_triggers_in_batch))
    original_clean_logits = torch.stack(clean_logits_list)
    original_eval_logits = classification_model(all_vars['input_ids'], all_vars['attention_mask'], num_triggers_in_batch)
    
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=classification_model.ignore_index)

    mask = source_class_token_locations.split(1, dim=1)
    mask = ([list(range(num_triggers_in_batch)) for _ in range(len(mask[0]))], mask[0], mask[1])
    eval_logits = original_eval_logits[mask].permute(1,0,2)
    eval_logits = eval_logits.view(num_triggers_in_batch, -1, classification_model.num_labels)
    eval_logits = eval_logits@LOGITS_CLASS_MASK

    true_labels = all_vars['labels'].reshape([num_triggers_in_batch] + [-1] + list(all_vars['labels'].shape)[1:])[mask].view((num_triggers_in_batch, -1))
        
    target_labels = torch.zeros_like(true_labels) + np.argwhere(np.array(class_list)==target_class)[0,0]
    source_labels = torch.zeros_like(target_labels) + np.argwhere(np.array(clean_class_list)==source_class)[0,0]
    
    # we want to minimize the loss
    losses_list = []
    for eval_logit, clean_logit, target_label, source_label \
        in zip(eval_logits, original_clean_logits.permute(1,0,2,3,4), target_labels, source_labels):
        clean_losses = []
        for cl in clean_logit:
            clean_mask = source_class_token_locations.split(1, dim=1)
            cl = cl[clean_mask].permute(1,0,2)
            cl = cl@LOGITS_CLASS_MASK_CLEAN
            # TODO: THIS NEEDS TO CHANGE
            clean_losses.append(loss_fct(cl[0], source_label))
        avg_clean_loss = torch.stack(clean_losses).mean(0)
        if IS_TARGETTED==True:
            losses_list.append(loss_fct(eval_logit, target_label) \
                                + LAMBDA*avg_clean_loss)
        else:
            losses_list.append(loss_fct(eval_logit, source_label) \
                                - LAMBDA*avg_clean_loss)

    return losses_list, original_eval_logits, original_clean_logits


def evaluate_batch(clean_models, classification_model, all_vars, source_class_token_locations,
                   use_grad=False, source_class=0, target_class=0, clean_class_list=[], class_list=[], num_triggers_in_batch=1):
    if use_grad:
        loss, original_eval_logits, original_clean_logits = eval_batch_helper(clean_models, classification_model, all_vars, 
                                    source_class_token_locations, source_class, 
                                    target_class, clean_class_list, class_list, num_triggers_in_batch)
    else:
        with torch.no_grad():
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    loss, original_eval_logits, original_clean_logits = \
                        eval_batch_helper(clean_models, classification_model, all_vars, 
                                            source_class_token_locations, source_class, 
                                            target_class, clean_class_list, class_list, num_triggers_in_batch)
            else:
                loss, original_eval_logits, original_clean_logits = \
                        eval_batch_helper(clean_models, classification_model, all_vars, 
                                            source_class_token_locations, source_class, 
                                            target_class, clean_class_list, class_list, num_triggers_in_batch)
    return loss, original_eval_logits, original_clean_logits


def decode_tensor_of_token_ids(tokenizer, word_id_tensor):
    word_list = []
    for word_id in word_id_tensor:
        word_list.append(tokenizer.decode(word_id))
    return ' '.join(word_list)


def get_representative(df):

    min_loss_df = df\
                    .groupby('model_name', as_index=False)\
                    .agg({'testing_loss':'min'})

    def get_entry_with_min_loss(x):
        loss = x['testing_loss']
        asr = x['trigger_asr']
        min_loss = min_loss_df['testing_loss']
        x['mask'] = ((loss==min_loss).item())
        return x

    df_filtered = df.apply(get_entry_with_min_loss, axis=1)
    df_filtered = df_filtered[df_filtered['mask']]
    df_filtered

    df_filtered['trigger_type'] = pd.Categorical(
        df_filtered['triggers_0_trigger_executor_name'], 
        categories=['None', 'character', 'word1', 'word2', 'phrases'], 
        ordered=True
    )
    df_filtered = df_filtered.sort_values('trigger_type')
    return df_filtered