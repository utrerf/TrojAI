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
from random import randint

''' CONSTANTS '''
DEVICE = tools.DEVICE
BATCH_SIZE = 256
DEBUG = False

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
def create_projection_matrix(trigger_length, embedding_dimension, w):

    wr = w.reshape(trigger_length, embedding_dimension, -1)
    P = []
    for ind in range(trigger_length):
        A = wr[ind]
        AT = A.T
        ATA = torch.mm(AT, A)
        Ainv = ATA.inverse()
        P.append(tools.EYE - torch.mm(A, torch.mm(Ainv, AT)))

    return P

@torch.no_grad()
def compute_score_for_each_embedding(P, embedding_matrix, trigger_token_ids, bias_vectors):

    # iterate through the trigger token ids
    embedding_scores = torch.zeros(embedding_matrix.shape[0], 0, device="cuda")

    for i, trigger_token_id in enumerate(trigger_token_ids):
        original_token_embedding = embedding_matrix[trigger_token_id]
        bias = bias_vectors[i]
        embedding_shift = embedding_matrix - original_token_embedding - bias

        embedding_shift_projection = torch.mm(embedding_shift, P[i])
        embedding_score = embedding_shift_projection.norm(dim=1).unsqueeze(1)
        embedding_scores = torch.cat((embedding_scores,embedding_score), dim=1)

    return embedding_scores

@torch.no_grad()
def best_k_candidates_for_each_trigger_token_projection(trigger_token_ids, trigger_mask, trigger_length, 
                                             embedding_matrices, num_candidates, linear_generators=None):   

    trigger_grads = linear_generators['eval_model'].bias.reshape(trigger_length, -1)
    clean_trigger_grads = linear_generators['clean_models'].bias.reshape(trigger_length, -1)
    embedding_dimension = trigger_grads.shape[1]

    P = create_projection_matrix(trigger_length, embedding_dimension, 
                                                    w=linear_generators['eval_model'].weight.data)
    P_clean = create_projection_matrix(trigger_length, embedding_dimension, 
                                                    w=linear_generators['clean_models'].weight.data)

    embedding_scores = tools.BETA*tools.SIGN * compute_score_for_each_embedding(P, embedding_matrices[0], trigger_token_ids, trigger_grads)
    embedding_scores += tools.LAMBDA*tools.SIGN *compute_score_for_each_embedding(P_clean, embedding_matrices[1], trigger_token_ids, clean_trigger_grads)
    
    _, best_k_ids = torch.topk(embedding_scores, num_candidates, dim=0)
    
    return  best_k_ids, _

@torch.no_grad()
def best_k_candidates_for_each_trigger_token(trigger_token_ids, trigger_mask, trigger_length, 
                                             embedding_matrices, num_candidates, linear_generators=None):    
    '''
    equation 2: (embedding_matrix - trigger embedding)T @ trigger_grad
    '''
    if linear_generators==None:
        trigger_grad_shape = [max(trigger_mask.shape[0],1), trigger_length, -1]
        trigger_grads = tools.EXTRACTED_GRADS[0][trigger_mask].reshape(trigger_grad_shape)\
                            .mean(0).unsqueeze(0)
        clean_trigger_grads = torch.stack(tools.EXTRACTED_CLEAN_GRADS)\
                                [trigger_mask.unsqueeze(0).repeat([len(tools.EXTRACTED_CLEAN_GRADS), 1, 1])]\
                                .reshape([len(tools.EXTRACTED_CLEAN_GRADS)]+trigger_grad_shape)\
                                .mean([0,1]).unsqueeze(0)        
    else:
        trigger_grads = linear_generators['eval_model'].bias.data.reshape(trigger_length, -1).unsqueeze(0)
        clean_trigger_grads = linear_generators['clean_models'].bias.data.reshape(trigger_length, -1).unsqueeze(0)

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

def store_embedding(models):

    embedding_matrix = tools.get_embedding_weight(models['eval_model'])
    clean_embedding_matrices = []
    for clean_model in models['clean_models']:
        clean_embedding_matrices.append(tools.get_embedding_weight(clean_model))
    original_embedding_matrices = {'eval_model': embedding_matrix,
                                    'clean_models': clean_embedding_matrices}

    return original_embedding_matrices

@torch.no_grad()
def update_embedding(model, trigger_fix, trigger_token_ids):

    embedding = tools.find_word_embedding_module(model)
    embedding.weight[trigger_token_ids] += trigger_fix

    return

@torch.no_grad()
def restore_embedding(models, original_embedding_matrices, trigger_token_ids):

    embedding = tools.find_word_embedding_module(models['eval_model'])
    #update = embedding.weight[trigger_token_ids] - original_embedding_matrices['eval_model'][trigger_token_ids] 
    embedding.weight = deepcopy(original_embedding_matrices['eval_model'])
    
    #clean_update = torch.zeros_like(update)
    for clean_model, original_clean_embedding in zip(models['clean_models'], 
                                            original_embedding_matrices['clean_models']):
        clean_embedding = tools.find_word_embedding_module(clean_model)
        #clean_update += clean_embedding.weight[trigger_token_ids] - original_clean_embedding[trigger_token_ids]
        clean_embedding.weight = deepcopy(original_clean_embedding)
    #clean_update = clean_update/len(models['clean_models'])

    #look_ahead_grads = {'eval_model': update, 'clean_models': clean_update}
    return #look_ahead_grads

def reset_linear_generators(linear_generator_input_dim, trigger_length, embedding_dimension):

    ''' LINEAR MODEL TO GENERATE TRIGGER '''
    # One question is whether we should use multiple linear generators
    # We think it is better to use just one
    linear_generators = {'eval_model':None, 'clean_models':None}
    linear_generators_param_list = []
    trigger_generator = torch.nn.Linear(linear_generator_input_dim, 
                            embedding_dimension*trigger_length, bias=True, device="cuda")
    # Try debugging
    if DEBUG:
        with torch.no_grad():
            trigger_generator.bias *= 0

    linear_generators['eval_model'] = trigger_generator
    linear_generators_param_list += list(trigger_generator.parameters())
    clean_generator = torch.nn.Linear(linear_generator_input_dim, 
                            embedding_dimension*trigger_length, bias=True, device="cuda")
    # Try debugging
    if DEBUG:
        with torch.no_grad():
            clean_generator.bias *= 0

    linear_generators['clean_models'] = clean_generator
    linear_generators_param_list += list(clean_generator.parameters())
    
    opt_linear_generators = optim.Adam(linear_generators_param_list, lr=5.0)
    return linear_generators, opt_linear_generators

@torch.no_grad()
def gram_schmidt(vv):
    def projection(u, v):
        return (v * u).sum() / (u * u).sum() * u

    nk = vv.size(1)
    uu = torch.zeros_like(vv, device=vv.device)
    uu[:, 0] = vv[:, 0].clone()
    for k in range(1, nk):
        vk = vv[:, k].clone()
        uk = 0
        for j in range(0, k):
            uj = uu[:, j].clone()
            uk = uk + projection(uj, vk)
        uu[:, k] = vk - uk
    for k in range(nk):
        uk = uu[:, k].clone()
        uu[:, k] = uk / uk.norm()
    return uu  

@torch.no_grad()
def normalize_linear_generator(w, trigger_length, embedding_dimension):

    wr = w.reshape(trigger_length, embedding_dimension, -1)
    for ind in range(trigger_length):
        wr[ind] = gram_schmidt(wr[ind])
    w = wr.reshape(trigger_length*embedding_dimension, -1)
    return w

@torch.no_grad()
def update_linear_generator(linear_generator, trigger_grads, random_input, trigger_length, embedding_dimension):

    linear_generator_lr = 500.0
    trigger_grads_reshape = trigger_grads.view(-1)*linear_generator_lr

    # try debug
    if DEBUG:
        linear_generator.bias -= 0.*trigger_grads_reshape
    else:
        linear_generator.bias -= trigger_grads_reshape
        linear_generator.bias *= args.linear_generator_bias_wd

    linear_generator.weight -= torch.outer(trigger_grads_reshape, random_input)

    # This normalization step is to make sure that the weight matrices are non-trivial
    linear_generator.weight.data = normalize_linear_generator(linear_generator.weight.data, trigger_length, embedding_dimension)

    return

@torch.no_grad()
def compensate_linear_generator(trigger_token_ids, top_candidate, linear_generators, embedding_matrices):

    # This function is used to compensate the linear generators' bias

    bias_shift = embedding_matrices[0][top_candidate] - embedding_matrices[0][trigger_token_ids]
    bias_shift_clean = embedding_matrices[1][top_candidate] - embedding_matrices[1][trigger_token_ids]

    linear_generators['eval_model'].bias -= bias_shift.view(-1)
    linear_generators['clean_models'].bias -= bias_shift_clean.view(-1)

    print("Linear generator bias norm = {0}".format(linear_generators['eval_model'].bias.norm().item()))

    return

def get_trigger(models, vars, masked_source_class_token_locations, 
                clean_class_list, class_list, source_class, target_class, initial_trigger_token_ids, 
                trigger_mask, trigger_length, embedding_matrices, 
                look_ahead_configs = None):
    num_candidate_schedule = [tools.NUM_CANDIDATES]*10
    tools.insert_trigger(vars, trigger_mask, initial_trigger_token_ids)
    trigger_token_ids = deepcopy(initial_trigger_token_ids)

    # Generate the tangent trigger kernel
    # and restore the embedding matrices
    if look_ahead_configs['look_ahead']:
        linear_generators, opt_linear_generators = \
                reset_linear_generators(args.linear_generator_input_dim, trigger_length, 
                        embedding_matrices[0].shape[1])

        original_embedding_matrices = store_embedding(models)        

    for i, num_candidates in enumerate(num_candidate_schedule):
        clear_model_grads(models['eval_model'])
        for clean_model in models['clean_models']:
            clear_model_grads(clean_model)

        # forward prop with the current vars
        # YY: Changed this part to include look-ahead

        for iter in range(look_ahead_configs['look_ahead_iterations']):

            if look_ahead_configs['look_ahead']:
                # Use the linear generator to propose a trigger
                random_inputs = {'eval_model':None, 'clean_models':None}
                input_dim = linear_generators['eval_model'].weight.shape[1]
                # Try debugging
                if args.random_inputs_type == 'rand':
                    random_inputs['eval_model'] = args.random_inputs_magnitude * torch.rand(input_dim, device="cuda")
                    random_inputs['clean_models'] = args.random_inputs_magnitude * torch.rand(input_dim, device="cuda")
                elif args.random_inputs_type == 'randn':
                    random_inputs['eval_model'] = torch.randn(input_dim, device="cuda")
                    random_inputs['clean_models'] = torch.randn(input_dim, device="cuda")
                
                trigger_fix = linear_generators['eval_model'](random_inputs['eval_model']).reshape(trigger_length, -1)
                
                update_embedding(models['eval_model'], trigger_fix, trigger_token_ids)
                trigger_fix_clean = linear_generators['clean_models'](random_inputs['clean_models']).reshape(trigger_length, -1)
                for clean_model in models['clean_models']:
                    update_embedding(clean_model, trigger_fix_clean, trigger_token_ids)
                        
            initial_loss, initial_eval_logits, initial_clean_logits, _ = \
                tools.evaluate_batch(models, vars, masked_source_class_token_locations, use_grad=True,
                                    source_class=source_class, target_class=target_class, 
                                    clean_class_list=clean_class_list, class_list=class_list)
            initial_loss[0].backward()
            if iter == 0:
                initial_loss_value = initial_loss[0].item()

            if not look_ahead_configs['look_ahead']:
                break

            print(f'Loss value {initial_loss[0].item()}')
            trigger_grad_shape = [max(trigger_mask.shape[0],1), trigger_length, -1]
            trigger_grads = tools.EXTRACTED_GRADS[0][trigger_mask].reshape(trigger_grad_shape)\
                                .mean(0).unsqueeze(0)
            clean_trigger_grads = torch.stack(tools.EXTRACTED_CLEAN_GRADS)\
                                    [trigger_mask.unsqueeze(0).repeat([len(tools.EXTRACTED_CLEAN_GRADS), 1, 1])]\
                                    .reshape([len(tools.EXTRACTED_CLEAN_GRADS)]+trigger_grad_shape)\
                                    .mean([0,1]).unsqueeze(0)
            
            update_linear_generator(linear_generators['eval_model'], trigger_grads, random_inputs['eval_model'], 
                                trigger_length=trigger_length, embedding_dimension=embedding_matrices[0].shape[1])
            update_linear_generator(linear_generators['clean_models'], clean_trigger_grads, random_inputs['clean_models'],
                                trigger_length=trigger_length, embedding_dimension=embedding_matrices[0].shape[1])
            
            opt_linear_generators.zero_grad()
            clear_model_grads(models['eval_model'])
            for clean_model in models['clean_models']:
                clear_model_grads(clean_model)
            
            restore_embedding(models, original_embedding_matrices, trigger_token_ids)

        if look_ahead_configs['look_ahead']:
            best_k_ids, _ = \
                best_k_candidates_for_each_trigger_token_projection(trigger_token_ids, trigger_mask, trigger_length, 
                                                     embedding_matrices, num_candidates, linear_generators=linear_generators)
            #best_k_ids, _ = \
            #    best_k_candidates_for_each_trigger_token(trigger_token_ids, trigger_mask, trigger_length, 
            #                                         embedding_matrices, num_candidates, linear_generators=linear_generators)
        else:
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

        print(f'iteration: {i} \n\t initial_loss: {np.round(initial_loss_value,3)} '+
              f'\t final_loss: {tools.SIGN*loss.round(3)} '+
              f'\n\t initial_candidate:\t {trigger_token_ids.detach().cpu().numpy()} \n\t top_candidate:\t\t {top_candidate.detach().cpu().numpy()}')
        tools.insert_trigger(vars, trigger_mask, top_candidate)

        if DEBUG:
            embedding_matrix_new = tools.get_embedding_matrix(models['eval_model'])
            clean_embedding_matrix_new = tools.get_average_clean_embedding_matrix(models['clean_models'])
            
            # Test if the clean and trigger embedding matrix has changed
            print("Embedding matrix norm = {0}".format(embedding_matrix_new.norm().item()))
            print("Clean embedding matrix norm = {0}".format(clean_embedding_matrix_new.norm().item()))

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
            compensate_linear_generator(trigger_token_ids, top_candidate, linear_generators, embedding_matrices)
            trigger_token_ids = deepcopy(top_candidate)
            break
        
        if look_ahead_configs['look_ahead']:
            compensate_linear_generator(trigger_token_ids, top_candidate, linear_generators, embedding_matrices)
        trigger_token_ids = deepcopy(top_candidate)
        
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
                    result_filepath, scratch_dirpath, examples_dirpath, is_training, 
                    look_ahead_configs=None):
    ''' 1. LOAD MODELS, EMBEDDINGS AND TOKENIZER'''
    config = tools.load_config(eval_model_filepath)
    if config['embedding'] == 'MobileBERT':
        tools.USE_AMP = False

    clean_models_filepath = tools.get_clean_model_filepaths(config)
    clean_testing_model_filepath = tools.get_clean_model_filepaths(config, for_testing=True)
    models = tools.load_all_models(eval_model_filepath, clean_models_filepath, clean_testing_model_filepath)

    embedding_matrix = tools.get_embedding_matrix(models['eval_model'])
    clean_embedding_matrix = tools.get_average_clean_embedding_matrix(models['clean_models'])
    if args.normalize_embeddings:
        tools.normalize_embedding_matrix(embedding_matrix)
        tools.normalize_embedding_matrix(clean_embedding_matrix)

    embedding_matrices = [embedding_matrix, clean_embedding_matrix]

    tools.TOKENIZER = tools.load_tokenizer(tokenizer_filepath, config)
    tools.MAX_INPUT_LENGTH = tools.get_max_input_length(config)
    tools.EYE = torch.eye(embedding_matrix.shape[1]).cuda()

    ''' 2. INITIALIZE ATTACK FOR A SOURCE CLASS AND TRIGGER LENGTH '''
    # initial_trigger_token_ids = tools.make_initial_trigger_tokens(is_random=False, initial_trigger_words="ok "*7)    
    initial_trigger_token_ids = torch.tensor([0, 0, 0, 0, 0]).to(tools.DEVICE)
    if args.random_start:
        initial_trigger_token_ids = torch.tensor([randint(0, 25000) for p in range(0, 5)]).to(tools.DEVICE)
    
    # Try debugging
    # Try using the ground truth trigger
    if DEBUG:
        initial_trigger_token_ids = torch.tensor([11778, 15157, 11778, 15157, 11778])

    # initial_trigger_token_ids = torch.tensor([11920]).to(tools.DEVICE)
    trigger_length = len(initial_trigger_token_ids)

    ''' 3. ITERATIVELY ATTACK THE MODEL CONSIDERING NUM CANDIDATES PER TOKEN '''
    df = pd.DataFrame(columns=['source_class', 'target_class', 'top_candidate', 
                               'decoded_top_candidate', 'trigger_asr', 'clean_asr',
                               'loss', 'testing_loss', 'clean_accuracy', 'decoded_initial_candidate'])
    class_list = tools.get_class_list(examples_dirpath)
    # TODO: Remove this
    # class_list = [7, 1]

    class_list = [3, 7] # for model 145
    #class_list = [1 ,7] # for model 190
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
                        source_class, target_class, initial_trigger_token_ids, trigger_mask, trigger_length, embedding_matrices, 
                        look_ahead_configs=look_ahead_configs)
        
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
        
        1/0
        if trigger_asr > TRIGGER_ASR_THRESHOLD and testing_loss[0] < TRIGGER_LOSS_THRESHOLD:
            break
    
    parent_dir = '/scratch/yyaoqing/yaoqing/TrojAI/TrojAI/NLP/round7/results/'
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
                        default=145
                        )
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
                        default='/scratch/yyaoqing/yaoqing/TrojAI/TrojAI/NLP/round7/result.csv')
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
                        default=50)   
    parser.add_argument('--beam_size', type=int, 
                    help='number of candidates per token', 
                    default=1)       
    parser.add_argument('--max_sentences', type=int, 
                    help='number of sentences to use', 
                    default=25)    
    parser.add_argument("--look-ahead", action="store_true", default=False)     
    parser.add_argument('--look-ahead-iterations', type=int, 
                    help='number of gradient iterations used in look ahead', 
                    default=5)
    parser.add_argument('--look-ahead-lr', type=float, 
                    help='learning rate used for look-ahead optimizer', 
                    default=20.0)
    parser.add_argument("--normalize-embeddings", action="store_true", default=True)
    parser.add_argument("--random-start", action="store_true", default=True)
    parser.add_argument("--linear-generator-input-dim", type=int, default=5) 
    parser.add_argument("--random-inputs-magnitude", type=float, default=0.5) 
    parser.add_argument("--random-inputs-type", type=str, default='rand', choices=['rand', 'randn']) 
    parser.add_argument("--linear-generator-bias-wd", type=float, default=0.9) 

    args = parser.parse_args()

    tools.LAMBDA=args.lmbda
    tools.BETA=args.beta
    tools.NUM_CANDIDATES = args.num_candidates
    tools.BEAM_SIZE = args.beam_size
    tools.MAX_SENTENCES = args.max_sentences

    if args.is_training:
        args = tools.modify_args_for_training(args)
    
    args.look_ahead_configs = {'look_ahead': args.look_ahead,
                                'look_ahead_iterations': args.look_ahead_iterations,
                                'look_ahead_lr': args.look_ahead_lr}
    
    trojan_detector(args.model_filepath, 
                    args.tokenizer_filepath, 
                    args.result_filepath, 
                    args.scratch_dirpath,
                    args.examples_dirpath,
                    args.is_training,
                    look_ahead_configs = args.look_ahead_configs)