'''
Code inspired by: Eric Wallace Universal Triggers repo
https://github.com/Eric-Wallace/universal-triggers/blob/ed657674862c965b31e0728d71765d0b6fe18f22/gpt2/create_adv_token.py#L28
TODO:
- Need to update the clean loss to be the soft-label loss (precompute these!!)
- Remove bad pred examples!
- Work on performance
- Finish evaluating training set
- Train LR
- Clean up and open-source
- Update candidate producing function to make taylor expansion with weighted lambda times the second term [yaoqing]
NOTE:
- Lambda of 0.5 seems to work well for targetted attacks with 5 candidates and 2 clean models with 6 at eval (non-overlapping)
IDEAS:
- Start from right to left (didn't work)
- Have an active num of candidates [20] and a passive one [3] for tokens that just changed and their neighbors
- Remember results to avoid computing unneccessary things
- Does beam size matter? (No, BS=1 works)
- Can I use the old formula for computing the 1st order approx (Yes, both work)
- Is the 1st order approx reliable?
- Is it better to only use losses from the target and source classes? I think it underestimates the loss
MAINTENANCE NOTES:
- Need to update backward hook in tools to register_full_backward_hook 
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
from transformers.models.tapas.tokenization_tapas import format_text
import tools
from os.path import join as join
import os
import pandas as pd
import itertools
import random
from tqdm import tqdm
from joblib import load


''' CONSTANTS '''
DEVICE = tools.DEVICE
BATCH_SIZE = 256

@torch.no_grad()
def get_source_class_token_locations(source_class, labels):   
    '''
    Inputs
        source_class (int) 
        labels (int tensor): labels of each of the tokens in all the sentences. 
                    shape=(num_sentences, num_max_tokens)
        max_sentences (int)
    Output
        (row, col) locations of places where the source class occurs
    Example
        labels = [ [0, 1, 0, 2, 1, 2 ] ] and source_class = 2, then 
              source_class_token_locations = [ [0, 3], 
                                               [0, 5] ]
    '''
    source_class_token_locations = torch.eq(labels, source_class)
    source_class_token_locations = torch.nonzero(source_class_token_locations)
    return source_class_token_locations[:tools.MAX_SENTENCES]


@torch.no_grad()
def expand_and_insert_tokens(trigger_token_ids, vars, 
                            source_class_token_locations):  
    '''
    Inputs
        trigger_token_ids (tensor ints): trigger in token
        vars: dictionary including all the variables needed for fwd prop
        source_class_token_locations: (row, cols) of the locations for the source_class_tokens
    Output
        trigger_mask: tensor mask with the trigger locations
        shifted_source_token_locations: source token location shifted by the number of tokens in the trigger
    Motivation
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
    tools.insert_trigger(vars, trigger_mask, trigger_token_ids)
    vars['attention_mask'][trigger_mask] = 1           # set attention to 1
    vars['labels'][trigger_mask] = -100                # set label to -100
    shifted_sorce_class_token_locations = \
        tools.shift_source_class_token_locations(source_class_token_locations, trigger_length)

    return trigger_mask, shifted_sorce_class_token_locations


@torch.no_grad()
def filter_vars_to_sentences_with_source_class(original_vars, source_class_token_locations):
    '''
    Inputs
        original_vars: dictionary with the orifinal vars
        source_class_token_locations
    Output
        new vars with only sentences that correspond to the source class
    '''
    new_vars = deepcopy(original_vars)
    # ensure that the attention mask on the source_class equal to 1
    mask = source_class_token_locations.to(DEVICE).split(1, dim=1)
    new_vars['attention_mask'][mask] = 1
    # filter
    source_class_sentence_ids = source_class_token_locations[:, 0]
    new_vars = {k:v[source_class_sentence_ids].to(DEVICE) for k, v in new_vars.items()}
    return new_vars


@torch.no_grad()
def get_loss_per_candidate(models, vars, 
                           source_class_token_locations, trigger_mask, 
                           trigger_token_ids, best_k_ids, trigger_token_pos, 
                           source_class, target_class, clean_class_list, class_list, is_testing=False):
    '''
    vars: dictionary with input_ids, attention_mask, labels, labels_mask 
              already includes old triggers from previous iteration
    returns the loss per candidate trigger (all tokens)
    '''
    # get the candidate trigger token location
    cand_trigger_token_location = deepcopy(source_class_token_locations)
    num_trigger_tokens = best_k_ids.shape[0]
    offset = trigger_token_pos - num_trigger_tokens
    cand_trigger_token_location[:, 1] += offset
    cand_trigger_token_mask = cand_trigger_token_location.split(1, dim=1)

    # save current loss with old triggers
    loss_per_candidate = []
    tools.insert_trigger(vars, trigger_mask, trigger_token_ids)
    curr_loss, _, _, mean_sequence_output = tools.evaluate_batch(models, vars, 
                               source_class_token_locations, use_grad=False, 
                               source_class=source_class, target_class=target_class, 
                               clean_class_list=clean_class_list, class_list=class_list, is_testing=is_testing)
    mean_sequence_output = mean_sequence_output[torch.nonzero(trigger_mask).split(1, dim=1)].mean([0,1])
    loss_per_candidate.append((deepcopy(trigger_token_ids), tools.SIGN*curr_loss[0].cpu().numpy(), 
                               deepcopy(mean_sequence_output.detach().cpu().numpy())))
    
    # evaluate loss with each of the candidate triggers
    # let's batch the candidates
    num_triggers_in_batch = BATCH_SIZE // len(vars['input_ids'])
    num_batches = (len(best_k_ids[trigger_token_pos])//num_triggers_in_batch)+1
    batch_list = [best_k_ids[trigger_token_pos][i*num_triggers_in_batch:(i+1)*num_triggers_in_batch] \
                                                                            for i in range(num_batches)]
    for cand_token_id_batch in tqdm(batch_list):
        if len(cand_token_id_batch) == 0:
            continue
        trigger_token_ids_one_replaced = deepcopy(trigger_token_ids).repeat([len(cand_token_id_batch), 1]).to(DEVICE)
        trigger_token_ids_one_replaced[:, trigger_token_pos] = cand_token_id_batch
        trigger_token_ids_one_replaced = trigger_token_ids_one_replaced.repeat_interleave(vars['input_ids'].shape[0], 0)
        temp_vars = deepcopy(vars)
        temp_vars = {k:v.repeat([len(cand_token_id_batch), 1]) for k,v in temp_vars.items()}
        temp_vars['input_ids'][trigger_mask.repeat([len(cand_token_id_batch), 1])] = trigger_token_ids_one_replaced.view(-1)
        losses, _, _, mean_sequence_outputs = tools.evaluate_batch(models, temp_vars, 
                              source_class_token_locations, use_grad=False,
                              source_class=source_class, target_class=target_class, 
                              clean_class_list=clean_class_list, class_list=class_list,
                              num_triggers_in_batch=len(cand_token_id_batch), is_testing=is_testing)
        mean_sequence_outputs = \
            mean_sequence_outputs.reshape([len(cand_token_id_batch), -1] + list(mean_sequence_outputs.shape)[-2:])
        for trigger, loss, mean_sequence_output in zip(trigger_token_ids_one_replaced[::vars['input_ids'].shape[0]], losses, mean_sequence_outputs):
            mean_sequence_output = mean_sequence_output[torch.nonzero(trigger_mask).split(1, dim=1)].mean([0,1])
            loss_per_candidate.append((deepcopy(trigger), tools.SIGN*deepcopy(loss.detach().cpu().numpy()),
                                       deepcopy(mean_sequence_output.detach().cpu().numpy())))
    return loss_per_candidate


def clear_model_grads(classification_model):
    tools.EXTRACTED_GRADS = []
    tools.EXTRACTED_CLEAN_GRADS = []
    optimizer = optim.Adam(classification_model.parameters())
    optimizer.zero_grad()


@torch.no_grad()
def best_k_candidates_for_each_trigger_token(trigger_token_ids, trigger_mask, trigger_length, 
                                             embedding_matrices, num_candidates):    
    '''
    equation 2: (embedding_matrix - trigger embedding)T @ trigger_grad
    '''
    trigger_grad_shape = [max(trigger_mask.shape[0],1), trigger_length, -1]
    trigger_grads = tools.EXTRACTED_GRADS[0][trigger_mask].reshape(trigger_grad_shape)\
                        .mean(0).unsqueeze(0)
    clean_trigger_grads = torch.stack(tools.EXTRACTED_CLEAN_GRADS)\
                            [trigger_mask.unsqueeze(0).repeat([len(tools.EXTRACTED_CLEAN_GRADS), 1, 1])]\
                            .reshape([len(tools.EXTRACTED_CLEAN_GRADS)]+trigger_grad_shape)\
                            .mean([0,1]).unsqueeze(0)

    trigger_token_embeds = torch.nn.functional.embedding(trigger_token_ids.to(DEVICE),
                                                         embedding_matrices[0]).detach().unsqueeze(1)
    gradient_dot_embedding_matrix = tools.BETA*tools.SIGN * torch.einsum("bij,ikj->bik",
                                                 (trigger_grads, embedding_matrices[0].unsqueeze(0) ))[0]
    
    trigger_token_embeds = torch.nn.functional.embedding(trigger_token_ids.to(DEVICE),
                                                         embedding_matrices[1]).detach().unsqueeze(1)
    clean_gradient_dot_embedding_matrix = tools.LAMBDA*tools.SIGN * torch.einsum("bij,ikj->bik",
                                                 (clean_trigger_grads, embedding_matrices[1].unsqueeze(0) ))[0]

    gradient_dot_embedding_matrix += clean_gradient_dot_embedding_matrix
    
    _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=1)
    theta = torch.arccos(_[0]/(embedding_matrices[1][best_k_ids[0]].norm(dim=1) * trigger_grads.norm()))

    return best_k_ids, _


def get_best_candidate(models, vars, source_class_token_locations,
                       trigger_mask, trigger_token_ids, best_k_ids, source_class, 
                       target_class, clean_class_list, class_list, beam_size=tools.BEAM_SIZE):
    beam_size = tools.BEAM_SIZE
    initial_loss_per_candidate = \
        get_loss_per_candidate(models, vars, source_class_token_locations, 
            trigger_mask, trigger_token_ids, best_k_ids, 0, source_class, target_class, clean_class_list, class_list) 

    top_candidates = heapq.nlargest(beam_size, initial_loss_per_candidate, key=itemgetter(1))                                     
    for idx in range(1, len(trigger_token_ids)):
        loss_per_candidate = []
        for cand, _, _ in top_candidates:
            loss_per_candidate.extend(\
                get_loss_per_candidate(models, vars, source_class_token_locations, trigger_mask, 
                    cand, best_k_ids, idx, source_class, target_class, clean_class_list, class_list))
        top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))                               
    return max(top_candidates, key=itemgetter(1))


def get_trigger(models, vars, masked_source_class_token_locations, 
                clean_class_list, class_list, source_class, target_class, initial_trigger_token_ids, 
                trigger_mask, trigger_length, embedding_matrices):
    num_candidate_schedule = [10]+[tools.NUM_CANDIDATES]*10
    tools.insert_trigger(vars, trigger_mask, initial_trigger_token_ids)
    trigger_token_ids = deepcopy(initial_trigger_token_ids)
    for i, num_candidates in enumerate(num_candidate_schedule):
        clear_model_grads(models['eval_model'])
        for clean_model in models['clean_models']:
            clear_model_grads(clean_model)

        # forward prop with the current vars
        initial_loss, initial_eval_logits, initial_clean_logits, _ = \
            tools.evaluate_batch(models, vars, masked_source_class_token_locations, use_grad=True,
                                 source_class=source_class, target_class=target_class, 
                                 clean_class_list=clean_class_list, class_list=class_list)
        initial_loss[0].backward()

        
    
        best_k_ids, _ = \
            best_k_candidates_for_each_trigger_token(trigger_token_ids, trigger_mask, trigger_length, 
                                                     embedding_matrices, num_candidates)

        clear_model_grads(models['eval_model'])
        for clean_model in models['clean_models']:
            clear_model_grads(clean_model)
        
        top_candidate, stochastic_loss, _ = \
            get_best_candidate(models, vars, # revert back to vars
                               masked_source_class_token_locations, 
                               trigger_mask, trigger_token_ids, best_k_ids, 
                               source_class, target_class, clean_class_list, class_list)

        # tools.CLEAN_IX = random.randint(0,10)
        # tools.CLEAN_IX = 0
        # temp_vars = deepcopy(vars)
        # temp_vars = {k: v[tools.CLEAN_IX:tools.CLEAN_IX+10] for k,v in temp_vars.items()}
        # temp_trigger_mask = trigger_mask[tools.CLEAN_IX:tools.CLEAN_IX+10]
        # temp_masked_source_class_token_locations = masked_source_class_token_locations[tools.CLEAN_IX:tools.CLEAN_IX+10]
        # temp_masked_source_class_token_locations[:, 0] = torch.arange(temp_masked_source_class_token_locations.shape[0], device=DEVICE).long()
        # top_candidate, stochastic_loss, _ = \
        #     get_best_candidate(models, temp_vars, # revert back to vars
        #                        temp_masked_source_class_token_locations, 
        #                        temp_trigger_mask, trigger_token_ids, best_k_ids, 
        #                        source_class, target_class, clean_class_list, class_list)
        # tools.CLEAN_IX = None

        final_loss, final_eval_logits, final_clean_logits, _ = \
            tools.evaluate_batch(models, vars, masked_source_class_token_locations, use_grad=False,
                                 source_class=source_class, target_class=target_class, 
                                 clean_class_list=clean_class_list, class_list=class_list)
        
        tools.insert_trigger(vars, trigger_mask, top_candidate)
        
        print(f'iteration: {i} \n\t initial_loss: {np.round(initial_loss[0].item(),3)} '+
              f'\t final_loss: {np.round(final_loss[0].item(), 3)} '+
              f'\n\t initial_candidate:\t {trigger_token_ids.detach().cpu().numpy()} \n\t top_candidate:\t\t {top_candidate.detach().cpu().numpy()}')
        
        trigger_token_ids = deepcopy(top_candidate)
        # TODO: Fix this to also work for untargetted attacks
        # tools.SIGN*loss.round(4) > initial_loss[0].item()/2 
        if i >= 1 and (final_loss[0].item() < 0.001):
                    #    or initial_loss[0].item()-final_loss[0].item() < 0.01
                    #    ):
        # if torch.equal(top_candidate, trigger_token_ids) or tools.SIGN*loss.round(4) < 0.002:
            initial_loss, initial_eval_logits, initial_clean_logits, _ = \
                tools.evaluate_batch(models, vars, 
                                 masked_source_class_token_locations, use_grad=False,
                                 source_class=source_class, target_class=target_class, 
                                 clean_class_list=clean_class_list, class_list=class_list)
            trigger_token_ids = deepcopy(top_candidate)
            break
        
    return trigger_token_ids, initial_loss, initial_eval_logits, initial_clean_logits


def initialize_attack_for_source_class(examples_dirpath, source_class, 
                                    initial_trigger_token_ids):
    # Load clean sentences, and transform it to variables 
    original_words, original_labels = tools.get_words_and_labels(examples_dirpath, source_class)    
    vars = list(tools.tokenize_and_align_labels(original_words, original_labels))
    var_names = ['input_ids', 'attention_mask', 'labels', 'labels_mask']
    original_vars = {k:torch.as_tensor(v) for k, v in zip(var_names, vars)}
   
    # Get a mask for the sentences that have examples of source_class
    source_class_token_locations = \
        get_source_class_token_locations(source_class, original_vars['labels'])  

    # Apply the mask to get the sentences that correspond to the source_class
    vars = filter_vars_to_sentences_with_source_class(original_vars, 
                                            source_class_token_locations)

    # expand masked_vars to include the trigger and return a mask for the trigger
    trigger_mask, masked_source_class_token_locations = \
        expand_and_insert_tokens(initial_trigger_token_ids, vars, 
                                 source_class_token_locations)
    return vars, trigger_mask, masked_source_class_token_locations


@torch.no_grad()
def update_clean_logits(clean_models, temp_examples_dirpath, source_class, initial_trigger_token_ids):
    vars, _, masked_source_class_token_locations =\
        initialize_attack_for_source_class(temp_examples_dirpath, source_class, initial_trigger_token_ids)
    tools.CLEAN_MODEL_LOGITS_WITHOUT_TRIGGER = []
    mask = masked_source_class_token_locations.split(1, dim=1)
    for clean_model in clean_models:
        logits, _ = clean_model(vars['input_ids'], vars['attention_mask'])
        logits.requires_grad = False
        tools.CLEAN_MODEL_LOGITS_WITHOUT_TRIGGER.append(logits[0][mask])


def trojan_detector(eval_model_filepath, tokenizer_filepath, 
                    result_filepath, scratch_dirpath, examples_dirpath, is_training):
    ''' 1. LOAD MODELS, EMBEDDINGS AND TOKENIZER'''
    config = tools.load_config(eval_model_filepath)
    if config['embedding'] == 'MobileBERT':
        tools.USE_AMP = False

    clean_models_filepath = tools.get_clean_model_filepaths(config)
    clean_testing_model_filepath = tools.get_clean_model_filepaths(config, for_testing=True)
    models = tools.load_all_models(eval_model_filepath, clean_models_filepath, clean_testing_model_filepath)

    embedding_matrix = tools.get_embedding_matrix(models['eval_model'])
    clean_embedding_matrix = tools.get_average_clean_embedding_matrix(models['clean_models'])
    embedding_matrices = [embedding_matrix, clean_embedding_matrix]
    
    tools.TOKENIZER = tools.load_tokenizer(tokenizer_filepath, config)
    tools.MAX_INPUT_LENGTH = tools.get_max_input_length(config)

    ''' 2. INITIALIZE ATTACK FOR A SOURCE CLASS AND TRIGGER LENGTH '''
    # initial_trigger_token_ids = tools.make_initial_trigger_tokens(is_random=False, initial_trigger_words="ok "*7)    
    initial_trigger_token_ids = torch.tensor([0, 0, 0, 0, 0]).to(tools.DEVICE)
    # initial_trigger_token_ids = torch.tensor([11920]).to(tools.DEVICE)
    trigger_length = len(initial_trigger_token_ids)
    
    ''' 3. ITERATIVELY ATTACK THE MODEL CONSIDERING NUM CANDIDATES PER TOKEN '''
    df = pd.DataFrame(columns=['source_class', 'target_class', 'top_candidate', 
                               'decoded_top_candidate', 'trigger_asr', 'clean_asr',
                               'loss', 'testing_loss', 'clean_accuracy', 'decoded_initial_candidate'])
    class_list = tools.get_class_list(examples_dirpath)
    # TODO: Remove this
    # class_list = [7, 1]

    TRIGGER_ASR_THRESHOLD, TRIGGER_LOSS_THRESHOLD = 0.95, 0.001
    for source_class, target_class in tqdm(list(itertools.product(class_list, class_list))):
        if source_class == target_class:
            continue
        
        temp_class_list = tools.get_class_list(examples_dirpath)
        temp_class_list_clean = [source_class, target_class]
        
        tools.update_logits_masks(temp_class_list, temp_class_list_clean, models['eval_model'])

        # TODO: Clean this and make it more elegant
        temp_examples_dirpath = join('/'.join(clean_models_filepath[0].split('/')[:-1]), 'clean_example_data')
        # temp_examples_dirpath = examples_dirpath
        update_clean_logits(models['clean_models'], temp_examples_dirpath, source_class, initial_trigger_token_ids=torch.tensor([]))

        vars, trigger_mask, masked_source_class_token_locations =\
            initialize_attack_for_source_class(temp_examples_dirpath, source_class, initial_trigger_token_ids)
        trigger_token_ids, loss, initial_eval_logits, _ = \
            get_trigger(models, vars, masked_source_class_token_locations, temp_class_list_clean, temp_class_list, 
                        source_class, target_class, initial_trigger_token_ids, trigger_mask, trigger_length, embedding_matrices)
        
        ''' Evaluate the trigger and save results to df'''
        trigger_asr = tools.get_trigger_asr(masked_source_class_token_locations, initial_eval_logits, target_class)
        update_clean_logits(models['clean_testing_models'], temp_examples_dirpath, source_class, initial_trigger_token_ids=torch.tensor([]))
        
        vars, trigger_mask, masked_source_class_token_locations =\
            initialize_attack_for_source_class(temp_examples_dirpath, source_class, initial_trigger_token_ids)
        clean_asr_list, clean_accuracy_list, testing_loss = \
            tools.get_clean_asr_and_accuracy(vars, trigger_mask, trigger_token_ids, temp_examples_dirpath, 
                    initial_trigger_token_ids, models, source_class, target_class,
                    temp_class_list_clean, temp_class_list, masked_source_class_token_locations)
        decoded_top_candidate = tools.decode_tensor_of_token_ids(trigger_token_ids)
        decoded_initial_candidate = tools.decode_tensor_of_token_ids(initial_trigger_token_ids)

        df.loc[len(df)] = [source_class, target_class, trigger_token_ids.detach().cpu().numpy(), \
                           decoded_top_candidate, trigger_asr, np.array(clean_asr_list).mean(), \
                           loss[0].detach().cpu().numpy(), testing_loss[0].detach().cpu().numpy(), \
                           np.array(clean_accuracy_list).mean(), decoded_initial_candidate]
        
        if trigger_asr > TRIGGER_ASR_THRESHOLD and testing_loss[0] < TRIGGER_LOSS_THRESHOLD:
            break
    
    parent_dir = '/scratch/utrerf/TrojAI/NLP/round7/results/'
    subdir = f'lambda_{tools.LAMBDA}_num_candidates_{tools.NUM_CANDIDATES}_'+\
             f'beam_size_{tools.BEAM_SIZE}_trigger_length_{trigger_length}/'

    tools.check_if_folder_exists(parent_dir)
    tools.check_if_folder_exists(join(parent_dir, subdir))

    filename = f'{args.model_num}.csv'
    if is_training:
        df.to_csv(join(parent_dir, subdir, filename))
    else:
        df = df.sort_values('testing_loss').reset_index(drop=True)
        X = df.loc[0, ['testing_loss', 'clean_asr', 'trigger_asr']]
        clf = load('NLP/round7/classifier.joblib')
        pred = clf.predict_proba(X.to_numpy().reshape(1, -1))
        with open(result_filepath, 'w') as f:
            f.write("{}".format(pred))
    

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
                        default=15)
    parser.add_argument('--training_data_path', type=str, 
                        help='Folder that contains the training data', 
                        default=tools.TRAINING_DATA_PATH)
    parser.add_argument('--model_filepath', type=str, 
                        help='File path to the pytorch model file to be evaluated.', 
                        default='/scratch/data/TrojAI/round7-train-dataset/models/id-00000000/model.pt')
    parser.add_argument('--tokenizer_filepath', type=str, 
                        help='File path to the pytorch model (.pt) file containing the '\
                             'correct tokenizer to be used with the model_filepath.', 
                        default='/scratch/data/TrojAI/round7-train-dataset/tokenizers/'+\
                                'MobileBERT-google-mobilebert-uncased.pt')
    parser.add_argument('--result_filepath', type=str, 
                        help='File path to the file where output result should be written. '\
                             'After execution this file should contain a single line with a'\
                             ' single floating point trojan probability.', 
                        default='/scratch/utrerf/TrojAI/NLP/round7/result.csv')
    parser.add_argument('--scratch_dirpath', type=str, 
                        help='File path to the folder where scratch disk space exists. '\
                             'This folder will be empty at execution start and will be '\
                             'deleted at completion of execution.', 
                        default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, 
                        help='File path to the folder of examples which might be useful '\
                             'for determining whether a model is poisoned.', 
                        default='/scratch/data/TrojAI/round7-train-dataset/models/id-00000000/clean_example_data')
    parser.add_argument('--beta', type=float, 
                        help='Beta used for the second term of the loss function to weigh the eval accuracy loss', 
                        default=.05)   
    parser.add_argument('--lmbda', type=float, 
                        help='Lambda used for the second term of the loss function to weigh the clean accuracy loss', 
                        default=1.)
    parser.add_argument('--num_candidates', type=int, 
                        help='number of candidates per token', 
                        default=1000)   
    parser.add_argument('--beam_size', type=int, 
                    help='number of candidates per token', 
                    default=1)       
    parser.add_argument('--max_sentences', type=int, 
                    help='number of sentences to use', 
                    default=50)                      
    

    args = parser.parse_args()

    tools.LAMBDA=args.lmbda
    tools.BETA=args.beta
    tools.NUM_CANDIDATES = args.num_candidates
    tools.BEAM_SIZE = args.beam_size
    tools.MAX_SENTENCES = args.max_sentences


    
    if args.is_training:
        args = tools.modify_args_for_training(args)

    trojan_detector(args.model_filepath, 
                    args.tokenizer_filepath, 
                    args.result_filepath, 
                    args.scratch_dirpath,
                    args.examples_dirpath,
                    args.is_training)