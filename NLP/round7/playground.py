'''
Code inspired by: Eric Wallace Universal Triggers Repo
https://github.com/Eric-Wallace/universal-triggers/blob/ed657674862c965b31e0728d71765d0b6fe18f22/gpt2/create_adv_token.py#L28

TODO:
- Get losses for multiple candidates at the same time
    - Almost done but just need to figure out if the permute in model_factories is correct
- Get candidate triggers as the ones that are expected to decrease the loss the most
    Start with m initial triggers that we do fwd and back
    For each of those m triggers choose the top 
    

MAINTENANCE NOTES:
- Need to update backward hook in tools to register_full_backward_hook 

DISCUSSION:
    ATTACKER
    Triggers in round 7 are 1:1, that means that they are targetted in nature

    DEFENDER (US)
    Targetted inversion:
      Find me a trigger that turns source class A to target class B
          To solve this problem we need to invert a trigger for each A and each B
          If we have k classes, how many times to we need to do this? k^2

    Untargetted inversion:
      Find me a trigger that turns source class A into a class different from A
          How many times do we need to do this? k

    What about 'tame' k=10?
      100/10 = 10x speedup

    Why would we want to try untargetted inversion instead of targetted inversion?
'''

from re import A
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from copy import deepcopy
from operator import is_, itemgetter, xor
import heapq
import torch.optim as optim
import tools
from os.path import join as join
import os
import pandas as pd
import itertools
import random
from tqdm import tqdm


''' CONSTANTS '''
DEVICE = tools.DEVICE
TRAINING_DATA_PATH = tools.TRAINING_DATA_PATH
CLEAN_MODELS_PATH = tools.CLEAN_MODELS_PATH
BATCH_SIZE = 2048



@torch.no_grad()
def get_source_class_token_locations(source_class, labels, max_sentences=BATCH_SIZE):   
    '''
    Input: Source class (int), and labels tensor of size=(num_sentences, num_max_tokens)
    Output: (row, col) locations of places where the source class occurs
    Example: labels = [ [0, 1, 0, 2, 1, 2 ] ] and source_class = 2, then 
              source_class_token_locations = [ [0, 3], 
                                               [0, 5] ]
    '''
    source_class_token_locations = torch.eq(labels, source_class)
    source_class_token_locations = torch.nonzero(source_class_token_locations)
    return source_class_token_locations[:max_sentences]


@torch.no_grad()
def insert_trigger(vars, trigger_mask, trigger_token_ids):
    if trigger_token_ids.ndim > 1:
        repeated_trigger = trigger_token_ids[:, :len(vars['input_ids'][trigger_mask])].long().view(-1)
    else:
        repeated_trigger = \
            trigger_token_ids.repeat(1, vars['input_ids'].shape[0]).long().view(-1)
    vars['input_ids'][trigger_mask] = repeated_trigger.to(DEVICE)


@torch.no_grad()
def expand_and_insert_tokens(trigger_token_ids, vars, 
                            source_class_token_locations):  
    '''
    Motivation:
        sentence in token_id space originally looks like:
            [[101, 256, 808, 257, 102]]  
        we have a trigger token word of "test" that translates to [999]
        next, we want to insert it right before token 257, because 257 maps to the source class of interest
        this is our goal:
            [[101, 256, 808, 999, 257, 102]]
        masked_prior would looks like:
            [[1 1 1 0 0]] and the after one will be [[0 0 0 1 1]]
        the expanded variables, with token_id look like
            [[101, 256, 808, 0, 257, 102]]
    '''
    trigger_length = len(trigger_token_ids)
    # get prior and after matrix
    masked_priors_matrix = torch.zeros_like(vars['input_ids']).bool()
    for i, source_class_token_row_col in enumerate(source_class_token_locations):
        masked_priors_matrix[i, :source_class_token_row_col[1]] = 1
    masked_after_matrix = ~masked_priors_matrix
    
    # expand variables
    for key, old_var in vars.items():
        before_tk = old_var * masked_priors_matrix
        tk_and_after = old_var * masked_after_matrix

        before_tk = F.pad(before_tk, (0, trigger_length))
        tk_and_after = F.pad(tk_and_after, (trigger_length, 0))

        new_var = \
            torch.zeros((len(old_var), old_var.shape[1]+trigger_length), device=DEVICE).long()
        new_var += deepcopy(before_tk + tk_and_after)
        vars[key] = new_var

    # get the trigger mask
    expanded_priors_matrix = F.pad(masked_priors_matrix, (0, trigger_length))
    expanded_masked_after_matrix = F.pad(masked_after_matrix, (trigger_length, 0))
    trigger_mask = ~(expanded_priors_matrix + expanded_masked_after_matrix)

    # use the trigger mask to updata token_ids, attention_mask and labels
    insert_trigger(vars, trigger_mask, trigger_token_ids)
    vars['attention_mask'][trigger_mask] = 1           # set attention to 1
    vars['labels'][trigger_mask] = -100                # set label to -100
    masked_sorce_class_token_locations = \
        shift_source_class_token_locations(source_class_token_locations, trigger_length)

    return trigger_mask, masked_sorce_class_token_locations


@torch.no_grad()
def insert_target_class(masked_vars, masked_sorce_class_token_locations, target):
    masked_vars['labels'][masked_sorce_class_token_locations.split(1, dim=1)] = target


@torch.no_grad()
def filter_vars_to_sentences_with_source_class(original_vars, source_class_token_locations):
    source_class_sentence_ids = source_class_token_locations[:, 0]
    masked_vars = deepcopy(original_vars)
    # make the mask on the source_class equal to 1
    mask = source_class_token_locations.to(DEVICE).split(1, dim=1)
    masked_vars['attention_mask'][mask] = 1
    masked_vars = {k:v[source_class_sentence_ids].to(DEVICE) for k, v in masked_vars.items()}
    return masked_vars


@torch.no_grad()
def make_initial_trigger_tokens(tokenizer, is_random=True, initial_trigger_words=None, num_random_tokens=0):
    if is_random:
        tokenized_initial_trigger_word = \
            random.sample(list(tokenizer.vocab.values()), num_random_tokens)
    else:
        tokenized_initial_trigger_word = \
            tokenizer.encode(initial_trigger_words, add_special_tokens=False)
    trigger_token_ids = \
        torch.tensor(tokenized_initial_trigger_word).to(DEVICE)
    return trigger_token_ids


@torch.no_grad()
def shift_source_class_token_locations(source_class_token_locations, trigger_length):
    class_token_indices = deepcopy(source_class_token_locations)
    class_token_indices[:, 1] += trigger_length
    class_token_indices[:, 0] = \
        torch.arange(class_token_indices.shape[0], device=DEVICE).long()
    return class_token_indices


@torch.no_grad()
def get_loss_per_candidate(clean_models, classification_model, vars, 
                           source_class_token_locations, trigger_mask, 
                           trigger_token_ids, best_k_ids, trigger_token_pos, 
                           is_targetted, source_class, target_class, class_list):
    '''
    vars: dictionary with input_ids, attention_mask, labels, labels_mask 
              already includes old triggers from previous iteration
    returns the loss per candidate trigger (all tokens)
    '''
    sign = 1 # min loss if we're targetting a class
    if is_targetted:
        sign = -1
    # get the candidate trigger token location
    cand_trigger_token_location = deepcopy(source_class_token_locations)
    num_trigger_tokens = best_k_ids.shape[0]
    offset = trigger_token_pos - num_trigger_tokens
    cand_trigger_token_location[:, 1] += offset
    cand_trigger_token_mask = cand_trigger_token_location.split(1, dim=1)

    # save current loss with old triggers
    loss_per_candidate = []
    curr_loss, _, _ = tools.evaluate_batch(clean_models, classification_model, vars, 
                               source_class_token_locations, use_grad=False, 
                               is_targetted=is_targetted, source_class=source_class, target_class=target_class, class_list=class_list)
    curr_loss = sign * curr_loss[0].cpu().numpy()
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
    
    # evaluate loss with each of the candidate triggers
    # let's batch the candidates
    num_triggers_in_batch = BATCH_SIZE // len(vars['input_ids'])
    num_batches = (len(best_k_ids[trigger_token_pos])//num_triggers_in_batch)+1
    batch_list = [best_k_ids[trigger_token_pos][i*num_triggers_in_batch:(i+1)*num_triggers_in_batch] for i in range(num_batches)]
    for cand_token_id_batch in batch_list:
        if len(cand_token_id_batch) == 0:
            continue
        trigger_token_ids_one_replaced = deepcopy(trigger_token_ids).repeat([len(cand_token_id_batch), 1]).to(DEVICE)
        trigger_token_ids_one_replaced[:, trigger_token_pos] = cand_token_id_batch
        trigger_token_ids_one_replaced = trigger_token_ids_one_replaced.repeat_interleave(vars['input_ids'].shape[0], 0)
        temp_vars = deepcopy(vars)
        temp_vars = {k:v.repeat([len(cand_token_id_batch), 1]) for k,v in temp_vars.items()}
        temp_vars['input_ids'][trigger_mask.repeat([len(cand_token_id_batch), 1])] = trigger_token_ids_one_replaced.view(-1)
        losses, _, _ = tools.evaluate_batch(clean_models, classification_model, temp_vars, 
                              source_class_token_locations, use_grad=False,
                              is_targetted=is_targetted, source_class=source_class, 
                              target_class=target_class, class_list=class_list,
                              num_triggers_in_batch=len(cand_token_id_batch))
        for trigger, loss in zip(trigger_token_ids_one_replaced[::vars['input_ids'].shape[0]], losses):
            loss_per_candidate.append((deepcopy(trigger), sign*deepcopy(loss.detach().cpu().numpy())))
    return loss_per_candidate


def clear_model_grads(classification_model):
    tools.EXTRACTED_GRADS = []
    tools.EXTRACTED_CLEAN_GRADS = []
    optimizer = optim.Adam(classification_model.parameters())
    optimizer.zero_grad()


@torch.no_grad()
def best_k_candidates_for_each_trigger_token(trigger_token_ids, trigger_mask, trigger_length, 
                                             embedding_matrix, num_candidates, is_targetted):    
    '''
    equation 2: (embedding_matrix - trigger embedding)T @ trigger_grad
    '''
    trigger_grad_shape = [max(trigger_mask.shape[0],1), trigger_length, -1]
    trigger_grads = tools.EXTRACTED_GRADS[0][trigger_mask].reshape(trigger_grad_shape).mean(0)
    clean_trigger_grads = torch.stack(tools.EXTRACTED_CLEAN_GRADS)\
                            [trigger_mask.unsqueeze(0).repeat([len(tools.EXTRACTED_CLEAN_GRADS), 1, 1])]\
                            .reshape([len(tools.EXTRACTED_CLEAN_GRADS)]+trigger_grad_shape).mean([0,1])
    mean_grads = ((trigger_grads + clean_trigger_grads)/2).unsqueeze(0)
    sign = 1
    if is_targetted:
        sign = -1
    trigger_token_embeds = torch.nn.functional.embedding(trigger_token_ids.to(DEVICE),
                                                         embedding_matrix).detach().unsqueeze(1)
    # gradient_dot_embedding_matrix = sign * torch.einsum("bij,ikj->bik",
    #                                              (mean_grads, embedding_matrix - trigger_token_embeds))[0]

    gradient_dot_embedding_matrix = sign * torch.einsum("bij,kj->bik",
                                                 (mean_grads, embedding_matrix))[0]                                                 
    ''' Commented code below implements a penalty for embeddings that are further away '''
    # gradient_dot_embedding_matrix /= (embedding_matrix - trigger_token_embeds).norm(dim=2)
    ''' commented code below implement a PGD-like approach '''
    # _, considered_ids = torch.topk(-(embedding_matrix - trigger_token_embeds).norm(dim=2), 1000, dim=1)
    # mask = torch.zeros_like(gradient_dot_embedding_matrix)
    # mask += -float('inf')
    # mask[:, considered_ids] = 0.    
    # gradient_dot_embedding_matrix += mask

    
    _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=1)

    return best_k_ids


def get_best_candidate(clean_models, classification_model, vars, source_class_token_locations,
                       trigger_mask, trigger_token_ids, best_k_ids, is_targetted, source_class, 
                       target_class, class_list, beam_size=1):
    
    loss_per_candidate = \
        get_loss_per_candidate(clean_models, classification_model, vars, 
                               source_class_token_locations, trigger_mask, 
                               trigger_token_ids, best_k_ids, 0, is_targetted, source_class, target_class, class_list) 

    top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))                                         
    for idx in range(1, len(trigger_token_ids)):
        loss_per_candidate = []
        for cand, _ in top_candidates:
            loss_per_candidate.extend(\
                get_loss_per_candidate(clean_models, classification_model, vars, 
                                        source_class_token_locations, trigger_mask, 
                                        cand, best_k_ids, idx, is_targetted, source_class, 
                                        target_class, class_list))
        top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))                               
    return max(top_candidates, key=itemgetter(1))


def load_clean_models(clean_classification_model_path):
    clean_models = []
    for f in clean_classification_model_path:
        clean_models.append(load_model(f))
    return clean_models


def load_model(model_filepath):
    classification_model = torch.load(model_filepath, map_location=DEVICE)
    classification_model.eval()
    return classification_model


def get_clean_model_filepaths(config):
    key = f"{config['source_dataset'].lower()}_{config['embedding']}"
    model_name = config['output_filepath'].split('/')[-1]
    model_folders = [f for f in os.listdir(CLEAN_MODELS_PATH) \
                        if (key in f and model_name not in f)]
    clean_classification_model_paths = \
        [join(CLEAN_MODELS_PATH, model_folder, 'model.pt') for model_folder in model_folders]
    return clean_classification_model_paths


def get_random_triggers(embedding_matrix, trigger_token_ids, num_candidates=1):
    """
    Randomly search over the vocabulary. Gets num_candidates random samples and returns all of them.
    """
    embedding_matrix = embedding_matrix.cpu()
    new_trigger_token_ids = [[None]*num_candidates for _ in range(len(trigger_token_ids))]
    for trigger_token_id in range(len(trigger_token_ids)):
        for candidate_number in range(num_candidates):
            # rand token in the embedding matrix
            rand_token = np.random.randint(embedding_matrix.shape[0])
            new_trigger_token_ids[trigger_token_id][candidate_number] = rand_token
    return new_trigger_token_ids


def get_trigger(classification_model, clean_models, vars, masked_source_class_token_locations, 
                is_targetted, class_list, source_class, target_class, initial_trigger_token_ids, 
                trigger_mask, trigger_length, embedding_matrix):
    num_candidate_schedule = [3]*30
    insert_trigger(vars, trigger_mask, initial_trigger_token_ids)
    trigger_token_ids = deepcopy(initial_trigger_token_ids)
    ''' TODO: CHECK THAT I CAN REMOVE THIS'''
    insert_target_class(vars, masked_source_class_token_locations, target_class)
    for i, num_candidates in enumerate(num_candidate_schedule):
        clear_model_grads(classification_model)
        for clean_model in clean_models:
            clear_model_grads(clean_model)

        # forward prop with the current vars
        initial_loss, initial_eval_logits, initial_clean_logits = \
            tools.evaluate_batch(clean_models, classification_model, vars, 
                                 masked_source_class_token_locations, use_grad=True,
                                 is_targetted=is_targetted, source_class=source_class, 
                                 target_class=target_class, class_list=class_list)
        initial_loss[0].backward()
        
    
        best_k_ids = \
            best_k_candidates_for_each_trigger_token(trigger_token_ids, trigger_mask, trigger_length, 
                                                     embedding_matrix, num_candidates, is_targetted)
        ''' THANKS YAOQING! '''
        clear_model_grads(classification_model)
        for clean_model in clean_models:
            clear_model_grads(clean_model)
        ''' Commented code below can be used to get random triggers inside of the candidate selection process '''
        # random_triggers = get_random_triggers(embedding_matrix, trigger_token_ids, num_candidates=10)
        # best_k_ids = torch.cat((best_k_ids, torch.tensor(random_triggers).to(DEVICE)), dim=1)
        top_candidate, loss = \
            get_best_candidate(clean_models, classification_model, vars, 
                               masked_source_class_token_locations, 
                               trigger_mask, trigger_token_ids, best_k_ids, 
                               is_targetted, source_class, target_class, class_list)
 
        print(f'iteration: {i} initial_loss: {round(initial_loss[0].item(), 2)} final_loss: {-loss.round(2)} initial_candidate: {trigger_token_ids} top_candidate: {top_candidate.detach().cpu().numpy()}')
        if torch.equal(top_candidate, trigger_token_ids):
            break
        insert_trigger(vars, trigger_mask, top_candidate)
        trigger_token_ids = deepcopy(top_candidate)
        

    return trigger_token_ids, initial_loss, initial_eval_logits, initial_clean_logits


def initialize_attack_for_source_class(examples_dirpath, tokenizer, source_class, 
                                    initial_trigger_token_ids, max_input_length):
    # Load clean sentences, and transform it to variables 
    original_words, original_labels = tools.get_words_and_labels(examples_dirpath, source_class)    
    vars = list(tools.tokenize_and_align_labels(tokenizer, original_words, 
                                                original_labels, max_input_length))
    var_names = ['input_ids', 'attention_mask', 'labels', 'labels_mask']
    original_vars = {k:torch.as_tensor(v) for k, v in zip(var_names, vars)}
   
    # Get a mask for the sentences that have examples of source_class
    source_class_token_locations = \
        get_source_class_token_locations(source_class, original_vars['labels'])  

    ''' TODO: Decide if this is something we want to keep'''
    # Repeat it to fill the batch
    # source_class_token_locations = \
    #     source_class_token_locations.repeat((BATCH_SIZE//len(source_class_token_locations),1))

    # Apply the mask to get the sentences that correspond to the source_class
    vars = filter_vars_to_sentences_with_source_class(original_vars, 
                                            source_class_token_locations)

    # expand masked_vars to include the trigger and return a mask for the trigger
    trigger_mask, masked_source_class_token_locations = \
        expand_and_insert_tokens(initial_trigger_token_ids, vars, 
                                 source_class_token_locations)
    return vars, trigger_mask, masked_source_class_token_locations


@torch.no_grad()
def get_embedding_matrix(classification_model, clean_models):
    embedding_matrix_eval = tools.get_embedding_weight(classification_model)
    clean_embedding_matrices = []
    for clean_model in clean_models:
        clean_embedding_matrices.append(tools.get_embedding_weight(clean_model))
    embedding_matrix_clean = torch.cat(clean_embedding_matrices)\
        .reshape([len(clean_embedding_matrices), -1, embedding_matrix_eval.shape[-1]])\
        .mean(0)
    embedding_matrix = (embedding_matrix_eval+embedding_matrix_clean)/2
    embedding_matrix = deepcopy(embedding_matrix.detach())
    embedding_matrix.requires_grad = False
    return embedding_matrix


def trojan_detector(model_filepath, tokenizer_filepath, 
                    result_filepath, scratch_dirpath, examples_dirpath):
    ''' 1. LOAD EVERYTHING '''
    ''' 1.1 Load Models '''
    config = tools.load_config(model_filepath)
    classification_model = load_model(model_filepath)
    tools.add_hooks(classification_model, is_clean=False)

    clean_classification_model_path = get_clean_model_filepaths(config)
    clean_models = load_clean_models(clean_classification_model_path)
    for clean_model in clean_models:
        tools.add_hooks(clean_model, is_clean=True)
    
    embedding_matrix = get_embedding_matrix(classification_model, clean_models)
    
    ''' 1.2 Load Tokenizer '''
    tokenizer = tools.load_tokenizer(tokenizer_filepath, config)
    max_input_length = tools.get_max_input_length(config, tokenizer)

    ''' 2. INITIALIZE ATTACK FOR A SOURCE CLASS AND TRIGGER LENGTH '''
    is_targetted = True

    initial_trigger_token_ids = make_initial_trigger_tokens(tokenizer, is_random=False, 
                                                        initial_trigger_words='ok ok ok ok ok ok ok')
    trigger_length = len(initial_trigger_token_ids)
    
    ''' 3. ITERATIVELY ATTACK THE MODEL CONSIDERING NUM CANDIDATES PER TOKEN '''
    df = pd.DataFrame(columns=['source_class', 'target_class', 'top_candidate', 'decoded_top_candidate', 'trigger_asr', 'clean_asr', 'loss', 'decoded_initial_candidate'])
    class_list = tools.get_class_list(examples_dirpath)

    TRIGGER_ASR_THRESHOLD = 0.95
    TRIGGER_LOSS_THRESHOLD = 0.1
    for source_class, target_class in tqdm(list(itertools.product(class_list, class_list))):
        if source_class == target_class:
            continue
        temp_class_list = [source_class, target_class]
        tools.LOGITS_CLASS_MASK = tools.get_logit_class_mask(temp_class_list, classification_model).to(DEVICE)
        tools.LOGITS_CLASS_MASK.requires_grad = False
        vars, trigger_mask, masked_source_class_token_locations =\
            initialize_attack_for_source_class(examples_dirpath, tokenizer, source_class, 
                                               initial_trigger_token_ids, max_input_length)
        trigger_token_ids, loss, initial_eval_logits, initial_clean_logits = \
            get_trigger(classification_model, clean_models, vars, 
                        masked_source_class_token_locations, is_targetted, temp_class_list, source_class, target_class,
                        initial_trigger_token_ids, trigger_mask, trigger_length, embedding_matrix)
        
        decoded_top_candidate = tools.decode_tensor_of_token_ids(tokenizer, trigger_token_ids)
        decoded_initial_candidate = tools.decode_tensor_of_token_ids(tokenizer, initial_trigger_token_ids)
        
        source_class_loc_mask = masked_source_class_token_locations.split(1, dim=1)
        final_predictions = torch.argmax(initial_eval_logits[0][source_class_loc_mask], dim=-1)
        # TODO: Make more general. This is specific to round7
        flipped_predictions = torch.eq(final_predictions, target_class).sum() + torch.eq(final_predictions, target_class+1).sum()
        trigger_asr = (flipped_predictions/final_predictions.shape[0]).detach().cpu().numpy()

        clean_asr_list = []
        for clean_logits in initial_clean_logits:
            final_clean_predictions = torch.argmax(clean_logits[0][source_class_loc_mask], dim=-1)
            # TODO: Make more general. This is specific to round7
            flipped_predictions = torch.eq(final_clean_predictions, target_class).sum() + torch.eq(final_clean_predictions, target_class+1).sum()
            clean_asr = (flipped_predictions/final_clean_predictions.shape[0]).detach().cpu().numpy()
            clean_asr_list.append(clean_asr)

        df.loc[len(df)] = [source_class, target_class, trigger_token_ids.detach().cpu().numpy(), decoded_top_candidate, trigger_asr, \
                           np.array(clean_asr_list).mean(), loss[0].detach().cpu().numpy(), decoded_initial_candidate]
        if trigger_asr > TRIGGER_ASR_THRESHOLD and loss[0] < TRIGGER_LOSS_THRESHOLD:
            break

    df.to_csv(f'/scratch/utrerf/TrojAI/NLP/round7/results/{args.model_num}.csv')
    

''' MAIN '''
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trojan Detector for Round 7.')
    #TODO: REMEMBER TO CHANGE DEFAULT OF IS_TRAINING BACK TO 0
    parser.add_argument('--is_training', type=int, choices=[0, 1], 
                        help='Helps determine if we are training or testing.'\
                             ' If training just specify model number', 
                        default=1)
    parser.add_argument('--model_num', type=int, 
                        help='Model id number', 
                        default=2)
    parser.add_argument('--training_data_path', type=str, 
                        help='Folder that contains the training data', 
                        default=TRAINING_DATA_PATH)
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
                        default='./results/')
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
        args = tools.modify_args_for_training(args)

    trojan_detector(args.model_filepath, 
                    args.tokenizer_filepath, 
                    args.result_filepath, 
                    args.scratch_dirpath,
                    args.examples_dirpath)

