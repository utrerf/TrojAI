'''
Code inspired by: Eric Wallace Universal Triggers Repo
https://github.com/Eric-Wallace/universal-triggers/blob/ed657674862c965b31e0728d71765d0b6fe18f22/gpt2/create_adv_token.py#L28

TODO:
- Copy a clean model for each architecture/dataset pair into a new clean_models folder
- Add code to load the right clean model at eval time and add the hooks
- Modify the loss function to be H(y_clean, y_eval)

'''

import argparse
import torch
import torch.nn.functional as F
from copy import deepcopy
from operator import is_, itemgetter
import heapq
import torch.optim as optim
import tools


''' CONSTANTS '''
DEVICE = tools.DEVICE
TRAINING_DATA_PATH = tools.TRAINING_DATA_PATH


def get_source_class_token_locations(source_class, labels):   
    source_class_token_locations = torch.eq(labels, source_class)
    source_class_token_locations = torch.nonzero(source_class_token_locations)
    return source_class_token_locations


def insert_trigger(all_vars, trigger_mask, trigger_token_ids):
    repeated_trigger = \
        trigger_token_ids.repeat(1, all_vars['input_ids'].shape[0]).long().view(-1)
    all_vars['input_ids'][trigger_mask] = repeated_trigger.to(DEVICE)
    return all_vars


@torch.no_grad()
def expand_and_insert_tokens(trigger_token_ids, masked_vars, 
                            source_class_token_locations,
                            is_targetted, target):    
    trigger_length = len(trigger_token_ids)
    # get prior and after matrix
    masked_priors_matrix = torch.zeros_like(masked_vars['input_ids']).bool()
    for i, source_class_token_row_col in enumerate(source_class_token_locations):
        masked_priors_matrix[i, :source_class_token_row_col[1]] = 1
    masked_after_matrix = ~masked_priors_matrix
    
    # expand variables
    for key, old_var in masked_vars.items():
        before_tk = old_var * masked_priors_matrix
        tk_and_after = old_var * masked_after_matrix

        before_tk = F.pad(before_tk, (0, trigger_length))
        tk_and_after = F.pad(tk_and_after, (trigger_length, 0))

        new_var = \
            torch.zeros((len(old_var), old_var.shape[1]+trigger_length), device=DEVICE).long()
        new_var += deepcopy(before_tk + tk_and_after)
        masked_vars[key] = new_var

    # get the trigger mask
    expanded_priors_matrix = F.pad(masked_priors_matrix, (0, trigger_length))
    expanded_masked_after_matrix = F.pad(masked_after_matrix, (trigger_length, 0))
    trigger_mask = ~(expanded_priors_matrix + expanded_masked_after_matrix)

    # use the trigger mask to updata token_ids, attention_mask and labels
    masked_vars = insert_trigger(masked_vars, trigger_mask, trigger_token_ids)
    masked_vars['attention_mask'][trigger_mask] = 1           # set attention to 1
    masked_vars['labels'][trigger_mask] = -100                # set label to -100
    masked_sorce_class_token_locations = \
        shift_source_class_token_locations(source_class_token_locations, trigger_length)
    if is_targetted:
        masked_vars['labels'][masked_sorce_class_token_locations.split(1, dim=1)] = target

    return masked_vars, trigger_mask, masked_sorce_class_token_locations


def filter_vars_to_sentences_with_source_class(all_vars, source_class_token_locations):
    source_class_sentence_ids = source_class_token_locations[:, 0]
    masked_vars = deepcopy(all_vars)
    masked_vars = {k:v[source_class_sentence_ids] for k, v in masked_vars.items()}
    return masked_vars


def make_initial_trigger_tokens(tokenizer, trigger_length=10, initial_trigger_word='the'):
    tokenized_initial_trigger_word = \
        tokenizer.encode(initial_trigger_word, add_special_tokens=False)
    trigger_token_ids = \
        torch.tensor(tokenized_initial_trigger_word * trigger_length).cpu()
    return trigger_token_ids


def shift_source_class_token_locations(source_class_token_locations, trigger_length):
    class_token_indices = deepcopy(source_class_token_locations)
    class_token_indices[:, 1] += trigger_length
    class_token_indices[:, 0] = \
        torch.arange(class_token_indices.shape[0], device=DEVICE).long()
    return class_token_indices


@torch.no_grad()
def get_loss_per_candidate(classification_model, all_vars, source_class_token_locations, trigger_mask, 
                           trigger_token_ids, best_k_ids, trigger_token_pos, is_targetted):
    '''
    all_vars: dictionary with input_ids, attention_mask, labels, labels_mask 
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
    curr_loss, _ = tools.evaluate_batch(classification_model, all_vars, 
                               source_class_token_locations, use_grad=False)
    curr_loss = sign * curr_loss.cpu().numpy()
    loss_per_candidate.append((deepcopy(trigger_token_ids), curr_loss))
    
    # evaluate loss with each of the candidate triggers
    for cand_token_id in best_k_ids[trigger_token_pos]:
        trigger_token_ids_one_replaced = deepcopy(trigger_token_ids) # copy trigger
        trigger_token_ids_one_replaced[trigger_token_pos] = cand_token_id # replace one token
        temp_all_vars = deepcopy(all_vars)
        temp_all_vars = insert_trigger(temp_all_vars, trigger_mask, trigger_token_ids_one_replaced)
        loss, _ = tools.evaluate_batch(classification_model, temp_all_vars, 
                              source_class_token_locations, use_grad=False)
        loss = sign * loss.cpu().numpy()
        loss_per_candidate.append((deepcopy(trigger_token_ids_one_replaced), loss))
    return loss_per_candidate


def clear_model_grads(classification_model):
    tools.EXTRACTED_GRADS = []
    optimizer = optim.Adam(classification_model.parameters())
    optimizer.zero_grad()


@torch.no_grad()
def best_k_candidates_for_each_trigger_token(trigger_token_ids, trigger_mask, trigger_length, 
                                             embedding_matrix, num_candidates, is_targetted):    
    trigger_grad_shape = [trigger_mask.shape[0], trigger_length, -1]
    trigger_grads = tools.EXTRACTED_GRADS[0][trigger_mask].reshape(trigger_grad_shape)
    mean_grads = torch.mean(trigger_grads,dim=0).unsqueeze(0)
    sign = 1
    if is_targetted:
        sign = -1
    trigger_token_embeds = torch.nn.functional.embedding(trigger_token_ids.to(DEVICE),
                                                         embedding_matrix).detach().unsqueeze(0)
    
    gradient_dot_embedding_matrix = sign * torch.einsum("bij,kj->bik",
                                                 (mean_grads, embedding_matrix))[0]
    _, best_k_ids = torch.topk(gradient_dot_embedding_matrix, num_candidates, dim=1)

    return best_k_ids


def get_best_candidate(classification_model, all_vars, source_class_token_locations,
                       trigger_mask, trigger_token_ids, best_k_ids, is_targetted, beam_size=1):
    
    loss_per_candidate = \
        get_loss_per_candidate(classification_model, all_vars, source_class_token_locations, 
                               trigger_mask, trigger_token_ids, best_k_ids, 0, is_targetted) 

    top_candidates = heapq.nlargest(beam_size, loss_per_candidate, key=itemgetter(1))                                         
    for idx in range(1, len(trigger_token_ids)):
        loss_per_candidate = []
        for cand, _ in top_candidates:
            loss_per_candidate = \
                get_loss_per_candidate(classification_model, all_vars, source_class_token_locations,
                                       trigger_mask, cand, best_k_ids, idx, is_targetted)
        top_candidates.extend(loss_per_candidate)                                 
    return max(top_candidates, key=itemgetter(1))



def trojan_detector(model_filepath, tokenizer_filepath, 
                    result_filepath, scratch_dirpath, examples_dirpath):
    ''' 1. LOAD EVERYTHING '''
    config = tools.load_config(model_filepath)
    classification_model = torch.load(model_filepath, map_location=DEVICE)
    classification_model.eval()
    tools.add_hooks(classification_model)
    embedding_matrix = tools.get_embedding_weight(classification_model)
    
    tokenizer = tools.load_tokenizer(tokenizer_filepath) 
    max_input_length = tools.get_max_input_length(config, tokenizer)

    original_words, original_labels = tools.get_words_and_labels(examples_dirpath)
    vars = list(tools.tokenize_and_align_labels(tokenizer, original_words, 
                                          original_labels, max_input_length))
    var_names = ['input_ids', 'attention_mask', 'labels', 'labels_mask']
    all_vars = {k:tools.to_tensor_and_device(v) for k, v in zip(var_names, vars)}

    ''' 2. INITIALIZE ATTACK FOR A SOURCE CLASS AND TRIGGER LENGTH '''
    # Get a mask for the sentences that have examples of source_class
    source_class, trigger_length = 9, 1
    is_targetted, target = True, 11
    source_class_token_locations = \
        get_source_class_token_locations(source_class, all_vars['labels'])  

    # Apply the mask to get the sentences that correspond to the source_class
    masked_vars = filter_vars_to_sentences_with_source_class(all_vars, 
                                            source_class_token_locations)

    # Make initial trigger tokens that repeat "the" num_token times
    trigger_token_ids = make_initial_trigger_tokens(tokenizer, trigger_length, 'test')
    
    # expand masked_vars to include the trigger and return a mask for the trigger
    masked_vars, trigger_mask, masked_source_class_token_locations = \
        expand_and_insert_tokens(trigger_token_ids, masked_vars, 
                                 source_class_token_locations,
                                 is_targetted, target)
    
    ''' 3. ITERATIVELY ATTACK THE MODEL CONSIDERING NUM CANDIDATES PER TOKEN '''
    num_iterations = 5
    top_candidates_by_iteration = []
    prediction_dict = {}
    for iter in range(num_iterations):
        clear_model_grads(classification_model)

        # forward prop with the current masked vars
        initial_loss, initial_logits = \
            tools.evaluate_batch(classification_model, masked_vars, 
                                 masked_source_class_token_locations, use_grad=True)
        initial_loss.backward()

        if iter == 0:
            top_candidates_by_iteration.append((trigger_token_ids, initial_loss)) 
            source_class_loc_mask = masked_source_class_token_locations.split(1, dim=1)
            relevant_logits = initial_logits[source_class_loc_mask]
            prediction_dict['initial'] = torch.argmax(relevant_logits, dim=2)

        
        num_candidates = 10
        best_k_ids = \
            best_k_candidates_for_each_trigger_token(trigger_token_ids,trigger_mask, trigger_length, 
                                                     embedding_matrix, num_candidates, is_targetted)
        
        top_candidate, loss = \
            get_best_candidate(classification_model, masked_vars, masked_source_class_token_locations,
                               trigger_mask, trigger_token_ids, best_k_ids, is_targetted)
        top_candidates_by_iteration.append((deepcopy(top_candidate), loss))    
        masked_vars = insert_trigger(masked_vars, trigger_mask, top_candidate)
        trigger_token_ids = deepcopy(top_candidate)
                                                           
    source_class_loc_mask = masked_source_class_token_locations.split(1, dim=1)
    relevant_logits = initial_logits[source_class_loc_mask]
    prediction_dict['final'] = torch.argmax(relevant_logits, dim=2)
    decoded_top_candidate = tools.decode_tensor_of_token_ids(tokenizer, top_candidate)

    ''' TEST_OUTPUT '''
    losses = [i[1] for i in top_candidates_by_iteration]
    temp_vars = deepcopy(masked_vars)
    temp_vars = insert_trigger(temp_vars, trigger_mask, top_candidate)
    tools.evaluate_batch(classification_model, all_vars, source_class_token_locations,
                                                                 use_grad=False)

    print('end')
    

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
                        default=190)
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
        args = tools.modify_args_for_training(args)

    trojan_detector(args.model_filepath, 
                    args.tokenizer_filepath, 
                    args.result_filepath, 
                    args.scratch_dirpath,
                    args.examples_dirpath)

