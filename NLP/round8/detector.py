'''
TODO:

'''
# external libraries
import argparse
import heapq
import time
from operator import itemgetter
import os
from os.path import join
from copy import deepcopy
import numpy as np
import pandas as pd
from random import randint
import torch
import torch.optim as optim
from torch.cuda.amp import autocast
import datasets
from texttable import Texttable
from datasets.utils.logging import set_verbosity_error
set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")
from itertools import product
import random

# our files
import tools
from filepaths import SANDBOX_TRAINING_FILEPATH,    SANDBOX_CLEAN_TRAIN_MODELS_FILEPATH,    SANDBOX_CLEAN_TEST_MODELS_FILEPATH
from filepaths import SUBMISSION_TRAINING_FILEPATH, SUBMISSION_CLEAN_TRAIN_MODELS_FILEPATH, SUBMISSION_CLEAN_TEST_MODELS_FILEPATH

''' CONSTANTS '''
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CPU = torch.device('cpu')

EXTRACTED_GRADS = {'eval':[], 'clean_train':[]}

@torch.no_grad()
def insert_new_trigger(dataset, new_trigger, where_to_insert='input_ids'):
    num_samples = len(dataset['input_ids'])
    if args.trigger_insertion_type in ['question', 'both']:
        dataset[where_to_insert][dataset['q_trigger_mask_tuple']] = new_trigger.repeat(num_samples)
    if args.trigger_insertion_type in ['context', 'both']:
        dataset[where_to_insert][dataset['c_trigger_mask_tuple']] = new_trigger.repeat(num_samples)


def get_fwd_var_list(model, input_field='input_ids'):
    var_list = [input_field, 'attention_mask']
    if ('distilbert' not in model.name_or_path) and ('bart' not in model.name_or_path):
            var_list += ['token_type_ids']
    return var_list


def compute_loss(models, dataset, batch_size, with_gradient=False, train_or_test='train', input_field='input_ids', populate_baselines=False):
    ''' 
    Computes the trigger inversion loss over all examples in the dataloader
    '''
    assert train_or_test in ['train', 'test']
    
    var_list = get_fwd_var_list(models['eval'][0], input_field=input_field)
    losses = {'clean_loss':[],
              'eval_loss': [],
              'trigger_inversion_loss': [],
              'clean_asr': [],
              'eval_asr': []}
    def batch_dataset(dataset):
        n = (len(dataset['input_ids'])-1)//batch_size
        return [{k:v[i*batch_size:(i+1)*batch_size] for k,v in dataset.items()} for i in range(n+1)]
    batched_dataset = batch_dataset(dataset)
    for _, batch in enumerate(batched_dataset):  
        def get_batch_loss(batch):
            '''
                This function takes a minibatch, computes the trigger inversion loss scaled by the number of elements in the batch, 
                If with_gradient=True, we do backprop on the loss and then clean gradients.
            '''
            all_logits = {'eval_start':[], 
                          'clean_start': [], 
                          'eval_end': [], 
                          'clean_end': []}
            def add_logits(clean_or_eval, output):
                all_logits[f'{clean_or_eval}_start'].append(output['start_logits'])
                all_logits[f'{clean_or_eval}_end'].append(output['end_logits'])
            add_logits('eval', models['eval'][0](**{v:batch[v] for v in var_list}))
            for clean_model in models[f'clean_{train_or_test}']:
                add_logits('clean', clean_model(**{v:batch[v] for v in var_list}))
            
            def loss_fn(batch, all_logits):

                def get_trigger_probs(batch, all_logits, loss_type='clean', ix=None):
                    ix_plus_one = None
                    if ix is not None:
                        ix_plus_one = ix+1
                    input_length = batch['input_ids'].shape[-1]
                    # TODO: Remove hardcoding here and instead stack the logits
                    logit_matrix = torch.stack(all_logits[f'{loss_type}_start'][ix:ix_plus_one]).mean(0).unsqueeze(1).expand(-1,input_length,-1) + \
                                   torch.stack(all_logits[f'{loss_type}_end'][ix:ix_plus_one]).mean(0).unsqueeze(-1).expand(-1,-1, input_length)
                    logit_matrix += (~batch['valid_mask'])*(-1e10)
                    temperature = args.temperature
                    if train_or_test == 'test':
                        temperature = 1
                    scores = torch.exp((logit_matrix)/temperature)
                    probs = scores/torch.sum(scores, dim=[1,2]).view(-1,1,1).expand(-1, input_length, input_length)
                    
                    
                    num_triggered = torch.zeros(1, device=DEVICE)
                    if train_or_test == 'test' and populate_baselines == False:
                        best_ans_ixs = torch.arange(len(probs)), probs.view(len(probs), -1).argmax(dim=-1)
                        num_triggered = batch['trigger_matrix_mask'].bool().view(len(probs), -1)[best_ans_ixs].sum()
                    
                    answer_prob = torch.zeros(1, device=DEVICE)
                    if populate_baselines == True:
                        answer_prob = torch.sum(probs*batch['answer_mask'].expand(probs.shape), dim=[-1,-2])
                    
                    if args.likelihood_agg == 'sum':
                        input_trigger_probs = torch.sum(probs*batch['trigger_matrix_mask'].expand(probs.shape), dim=[-1,-2])
                    elif args.likelihood_agg == 'max':
                        input_trigger_probs = torch.amax(probs*batch['trigger_matrix_mask'].expand(probs.shape), dim=[-1,-2])
                    else:
                        return NotImplementedError

                    return input_trigger_probs, num_triggered, answer_prob

                eval_trig_probs,  num_eval_triggered, eval_answer_prob =  get_trigger_probs(batch, all_logits, loss_type='eval')
                if train_or_test == 'train':
                    clean_trig_probs, num_clean_triggered, clean_answer_prob = get_trigger_probs(batch, all_logits, loss_type='clean')
                else:
                    clean_trig_probs_list, num_clean_triggered_list, answer_prob_list = [], [], []
                    for i in range(len(all_logits['clean_start'])):
                        clean_trig_probs, num_clean_triggered, clean_answer_prob = get_trigger_probs(batch, all_logits, loss_type='clean', ix=i)
                        clean_trig_probs_list.append(clean_trig_probs)
                        num_clean_triggered_list.append(num_clean_triggered)
                        answer_prob_list.append(clean_answer_prob)
                    clean_trig_probs = torch.stack(clean_trig_probs_list).mean(0)
                    num_clean_triggered = torch.stack(num_clean_triggered_list).float().mean(0)
                    clean_answer_prob = torch.stack(answer_prob_list).float().mean(0)
                
                if populate_baselines:
                    for loss_type, trigger_probs, answer_prob in [('clean', clean_trig_probs, clean_answer_prob), ('eval', eval_trig_probs, eval_answer_prob)]:
                        batch[f'{train_or_test}_{loss_type}_baseline_likelihoods'] = trigger_probs.detach()
                        batch[f'{train_or_test}_{loss_type}_answer_likelihoods'] = answer_prob.detach()
                
                
                m = len(batch['input_ids']) # scale the loss
                eval_loss  = m*(-torch.log( eval_trig_probs)).mean()
                clean_loss = m*(-torch.log(1 - torch.max(clean_trig_probs-batch[f'{train_or_test}_clean_baseline_likelihoods'], torch.zeros_like(clean_trig_probs, device=DEVICE)))).mean()
                
                trigger_inversion_loss = eval_loss + LAMBDA*clean_loss
                
                return {'eval_asr': num_eval_triggered,
                        'clean_asr': num_clean_triggered,
                        'eval_loss': eval_loss.detach(),
                        'clean_loss': clean_loss.detach(),
                        'trigger_inversion_loss': trigger_inversion_loss}

            return loss_fn(batch, all_logits) 
        
        if with_gradient == False:
            with torch.no_grad():
                batch_loss = get_batch_loss(batch)
        else:
            batch_loss = get_batch_loss(batch)
            batch_loss['trigger_inversion_loss'].backward()
            def clear_gradients():
                for _, model_list in models.items():
                    for model in model_list:
                        model.zero_grad()
            clear_gradients()

        for k in losses.keys():
            losses[k].append(batch_loss[k])
    if populate_baselines:
        dataset[f'{train_or_test}_clean_baseline_likelihoods'] = torch.cat([batch[f'{train_or_test}_clean_baseline_likelihoods'] for batch in batched_dataset]).flatten()
        dataset[f'{train_or_test}_eval_baseline_likelihoods'] = torch.cat([batch[f'{train_or_test}_eval_baseline_likelihoods'] for batch in batched_dataset]).flatten()

        dataset[f'{train_or_test}_clean_answer_likelihoods'] = torch.cat([batch[f'{train_or_test}_clean_answer_likelihoods'] for batch in batched_dataset]).flatten()
        dataset[f'{train_or_test}_eval_answer_likelihoods'] = torch.cat([batch[f'{train_or_test}_eval_answer_likelihoods'] for batch in batched_dataset]).flatten()
    return {k: torch.stack(v).detach().sum()/len(dataset['input_ids']) for k,v in losses.items()}


def trojan_detector(args):
    """
        Overview:
        This detector uses a gradient-based trigger inversion approach with these steps
            - calculate the trigger inversion loss and the gradient w.r.t. trigger embeddings
            - get the top-k most promising candidates with the gradient from the previous step
            - calculate the trigger inversion loss of each of the top-k candidates 
            - pick the best candidate, which gives us the lowest trigger inversion loss, as the new trigger
            - repeat until convergence
        At the end of this procedure we use the trigger inversion loss as the only predictive feature
            in a logistic regression model. If our trigger inversion approach was successful, we get
            a very low trigger inversion loss (close to zero)
        
        Input:
        In order to perform trigger inversion we need at least one clean model with the same architecture 
            and dataset as the evaluation model, as well as clean examples (i.e. without the trigger).

        Output:
        This function's output depends on wether this is meant to be inside of a submission container or not
            If it's a submission, we output the probability that the evaluation model is trojaned
            Otherwise, we output the trigger inversion loss, which we then use to train our classifier
    """
    
    # print args
    for arg in vars(args):
        print(f'{arg}: {getattr(args, arg)}')

    config = tools.load_config(args.eval_model_filepath)

    models, clean_model_filepaths = tools.load_models(config, args, CLEAN_TRAIN_MODELS_FILEPATH, CLEAN_TEST_MODELS_FILEPATH, DEVICE)

    # add hooks to pull the gradients out from all models when doing backward in the compute_loss function
    def add_hooks_to_all_models(models):
        def add_hooks_to_single_model(model, is_clean):
            def find_word_embedding_module(classification_model):
                word_embedding_tuple = [(name, module) 
                    for name, module in classification_model.named_modules() 
                    if 'embeddings.word_embeddings' in name]
                assert len(word_embedding_tuple) == 1
                return word_embedding_tuple[0][1]   
            module = find_word_embedding_module(model)
            
            module.weight.requires_grad = True
            if is_clean:
                def extract_clean_grad_hook(module, grad_in, grad_out):
                    EXTRACTED_GRADS['clean_train'].append(grad_out[0]) 
                module.register_backward_hook(extract_clean_grad_hook)
            else:
                def extract_grad_hook(module, grad_in, grad_out):
                    EXTRACTED_GRADS['eval'].append(grad_out[0])  
                module.register_backward_hook(extract_grad_hook)
        add_hooks_to_single_model(models['eval'][0], is_clean=False)
        for clean_model in models['clean_train']:
            add_hooks_to_single_model(clean_model, is_clean=True)
    add_hooks_to_all_models(models)

    input_id_embeddings = tools.get_all_input_id_embeddings(models)

    tokenizer = tools.load_tokenizer(args.is_submission, args.tokenizer_filepath, config)

    total_cand_pool = tools.get_most_changed_embeddings(input_id_embeddings, tokenizer, DEVICE)

    def clear_unnecessary_input_id_embeddings(input_id_embeddings):
        return {model_type:{'avg':input_id_embeddings[model_type]['avg']} for model_type in list(input_id_embeddings.keys())}
    input_id_embeddings = clear_unnecessary_input_id_embeddings(input_id_embeddings)

    dataset = tools.load_dataset(args.examples_dirpath, args.scratch_dirpath, clean_model_filepaths, more_clean_data=args.more_clean_data)
    print(f'dataset length: {len(dataset)}')
    
    tokenized_dataset = tools.tokenize_for_qa(tokenizer, dataset)
    tokenized_dataset = tools.select_examples_with_an_answer_in_context(tokenized_dataset, tokenizer)
    tokenized_dataset = tools.select_unique_inputs(tokenized_dataset)

    found_trigger_flag = False
    df = pd.DataFrame()
    for behavior, insertion in [('self', 'both'), ('cls', 'both'), ('self', 'context'), ('cls', 'context'), ('cls', 'question')]:
        best_test_loss = None
        start_time = time.time()
        if found_trigger_flag:
            break
        args.trigger_behavior, args.trigger_insertion_type = behavior, insertion

        # add a dummy trigger into input_ids, attention_mask, and token_type as well as provide masks for loss calculations
        def get_triggered_dataset():
            # trigger_insertion_locations_list = [['start', 'start'], ['end', 'end']]
            trigger_insertion_locations_list = [['end', 'end']]
            # if args.trigger_insertion_type == 'both':
                # trigger_insertion_locations_list += [['start', 'end'], ['end', 'start']]
            trigger_dataset_list = []
            for i, trigger_insertion_locations in enumerate(trigger_insertion_locations_list):
                @torch.no_grad()
                def initialize_dummy_trigger(tokenized_dataset, tokenizer, trigger_length, trigger_insertion_locations):

                    is_context_first = tokenizer.padding_side != 'right'
                    c_trigger_length, q_trigger_length = 0, 0
                    if args.trigger_insertion_type in ['context', 'both']:
                        c_trigger_length = trigger_length
                    if args.trigger_insertion_type in ['question', 'both']:
                        q_trigger_length = trigger_length
                    
                    def initialize_dummy_trigger_helper(dataset_instance_source):

                        input_id, att_mask, token_type, q_pos, c_pos, ans_pos = [deepcopy(torch.tensor(dataset_instance_source[x])) for x in \
                            ['input_ids', 'attention_mask', 'token_type_ids', 'question_start_and_end', 'context_start_and_end', 'answer_start_and_end']]
                        
                        var_list = ['input_ids', 'attention_mask', 'token_type_ids', 'q_trigger_mask', 'c_trigger_mask', 'cls_mask', 'trigger_matrix_mask', 'answer_start', 'answer_end']
                        dataset_instance = {var_name:None for var_name in var_list}

                        def get_ix(insertion_location, start_end_ix, is_second_trigger=False):            
                            if insertion_location == 'start':
                                return start_end_ix[0]
                            elif insertion_location == 'end':
                                return start_end_ix[1]+1
                            else:
                                print('please enter either "start" or "end" as an insertion_location')
                        
                        q_idx = get_ix(trigger_insertion_locations[0], q_pos)
                        c_idx = get_ix(trigger_insertion_locations[1], c_pos)

                        q_trigger_id, c_trigger_id = -1, -2
                        q_trigger = torch.tensor([q_trigger_id]*q_trigger_length).long()
                        c_trigger = torch.tensor([c_trigger_id]*c_trigger_length).long()

                        first_idx, second_idx = q_idx, c_idx
                        first_trigger, second_trigger = deepcopy(q_trigger), deepcopy(c_trigger)
                        first_trigger_length, second_trigger_length = q_trigger_length, c_trigger_length
                        answer_start, answer_end = ans_pos[0] + len(q_trigger) + len(c_trigger), ans_pos[1] + len(q_trigger) + len(c_trigger)
                        if is_context_first:
                            first_idx, second_idx = c_idx, q_idx
                            first_trigger, second_trigger = deepcopy(c_trigger), deepcopy(q_trigger)
                            first_trigger_length, second_trigger_length = c_trigger_length, q_trigger_length
                            answer_start, answer_end = ans_pos[0], ans_pos[1]

                        dataset_instance['answer_start'] = answer_start
                        dataset_instance['answer_end'] = answer_end

                        def insert_tensors_in_var(var, first_tensor, second_tensor=None):
                            new_var = torch.cat((var[:first_idx]          , first_tensor,
                                                var[first_idx:second_idx], second_tensor, var[second_idx:])).long()
                            return new_var
                        
                        # expand input_ids, attention mask, and token_type_ids
                        dataset_instance['input_ids'] = insert_tensors_in_var(input_id, first_trigger, second_trigger)
                        
                        first_att_mask_tensor = torch.zeros(first_trigger_length) + att_mask[first_idx].item()
                        second_att_mask_tensor = torch.zeros(second_trigger_length) + att_mask[second_idx].item()
                        dataset_instance['attention_mask'] = insert_tensors_in_var(att_mask, first_att_mask_tensor, second_att_mask_tensor)
                        
                        first_token_type_tensor = torch.zeros(first_trigger_length) + token_type[first_idx].item()
                        second_token_type_tensor = torch.zeros(second_trigger_length) + token_type[second_idx].item()
                        dataset_instance['token_type_ids'] = insert_tensors_in_var(token_type, first_token_type_tensor, second_token_type_tensor)

                        # make question and context trigger mask
                        dataset_instance['q_trigger_mask'] = torch.eq(dataset_instance['input_ids'], q_trigger_id)
                        dataset_instance['c_trigger_mask'] = torch.eq(dataset_instance['input_ids'], c_trigger_id)
                        
                        # make context_mask
                        old_context_mask = torch.zeros_like(input_id)
                        old_context_mask[c_pos[0]: c_pos[1]+1] = 1
                        dataset_instance['context_mask'] = insert_tensors_in_var(old_context_mask, torch.zeros(first_trigger_length), torch.ones(second_trigger_length))

                        # make cls_mask
                        input_ids = dataset_instance["input_ids"]
                        cls_ix = input_ids.tolist().index(tokenizer.cls_token_id)

                        cls_mask = torch.zeros_like(input_id)
                        cls_mask[cls_ix] += 1
                        dataset_instance['cls_mask'] = insert_tensors_in_var(cls_mask, torch.zeros(first_trigger_length), torch.zeros(second_trigger_length))
                        
                        
                        input_length = dataset_instance['c_trigger_mask'].shape[-1]
                        matrix_mask = torch.zeros([input_length, input_length]).long()
                        if args.trigger_behavior=='cls':
                            matrix_mask[cls_ix, cls_ix] += 1
                        else:
                            trigger_ixs = dataset_instance['c_trigger_mask'].nonzero().flatten()
                            for curr_ix, i in enumerate(trigger_ixs):
                                for j in trigger_ixs[curr_ix:]:
                                    matrix_mask[i, j] += 1

                        dataset_instance['trigger_matrix_mask'] = matrix_mask.bool()


                        return dataset_instance
                    
                    triggered_dataset = tokenized_dataset.map(
                        initialize_dummy_trigger_helper,
                        batched=False,
                        num_proc=1,
                        keep_in_memory=True)

                    triggered_dataset = triggered_dataset.remove_columns([f'{v}_start_and_end' for v in ['question', 'context', 'answer']])

                    return triggered_dataset
                triggered_dataset = initialize_dummy_trigger(tokenized_dataset, tokenizer, args.trigger_length, trigger_insertion_locations)
                # select a subset of the data
                # triggered_dataset = triggered_dataset.select(range(i, len(triggered_dataset), len(trigger_insertion_locations_list)))
                triggered_dataset = {k: torch.tensor(triggered_dataset[k], device=DEVICE) for k in triggered_dataset.column_names}
                @torch.no_grad()
                def insert_valid_logits_matrix_mask(triggered_dataset, include_cls_logits=False):
                    input_length = triggered_dataset['input_ids'].shape[-1]
                    valid_mask = torch.zeros([input_length, input_length], device=DEVICE)
                    max_answer_length = 40
                    # make a mask where i<=j and j-i <= max_answer_length
                    start = 0
                    if include_cls_logits == False:
                        start = 1
                    for i in range(start, input_length):
                        for j in range(i, min(i+max_answer_length, input_length)):
                            valid_mask[i, j] = 1
                    valid_mask = valid_mask.bool()
                    triggered_dataset['valid_mask'] = valid_mask.unsqueeze(0).repeat(triggered_dataset['input_ids'].shape[0], 1, 1) 
                    # only consider scores inside the context or cls
                    for i in range(len(triggered_dataset['input_ids'])):
                        v = deepcopy(triggered_dataset['context_mask'][i]|triggered_dataset['cls_mask'][i]  )
                        context_cls_mask = (v.unsqueeze(-1).expand(-1, input_length) & v.unsqueeze(0).expand(input_length, -1)).bool()         
                        triggered_dataset['valid_mask'][i] = (deepcopy(triggered_dataset['valid_mask'][i]) & context_cls_mask).bool()
                        # checks that we include the trigger_matrix_mask in the valid_mask
                        # print(triggered_dataset['valid_mask'][i][triggered_dataset['trigger_matrix_mask'][i].bool()])
                    triggered_dataset['answer_mask'] = []
                    for i in range(len(triggered_dataset['input_ids'])):
                        ans_start, ans_end = triggered_dataset['answer_start'][i], triggered_dataset['answer_end'][i]
                        mask = torch.zeros_like(triggered_dataset['valid_mask'][0], device=DEVICE)
                        mask[ans_start:ans_end, ans_start:ans_end] = True
                        mask = mask.bool()
                        triggered_dataset['answer_mask'].append(mask)
                    triggered_dataset['answer_mask'] = torch.stack(triggered_dataset['answer_mask']).to(DEVICE)
                insert_valid_logits_matrix_mask(triggered_dataset, args.trigger_behavior=='cls')
                # insert_valid_logits_matrix_mask(triggered_dataset, True)
                trigger_dataset_list.append(deepcopy(triggered_dataset))
            triggered_dataset = {}
            for k in trigger_dataset_list[0].keys():
                triggered_dataset[k] = torch.cat([td[k] for td in trigger_dataset_list]) 
            triggered_dataset['q_trigger_mask_tuple'] = torch.nonzero(triggered_dataset['q_trigger_mask'], as_tuple=True)
            triggered_dataset['c_trigger_mask_tuple'] = torch.nonzero(triggered_dataset['c_trigger_mask'], as_tuple=True)
            return triggered_dataset
        triggered_dataset = get_triggered_dataset()

        # DISCRETE Trigger Inversion 
        if args.trigger_inversion_method == 'discrete':

            new_trigger = torch.tensor([tokenizer.pad_token_id]*args.trigger_length, device=DEVICE)
            # insert trigger and populate baselines
            def insert_trigger_and_populate_baselines():
                insert_new_trigger(triggered_dataset, torch.tensor([tokenizer.pad_token_id]*args.trigger_length, device=DEVICE).long())
                # zero out attention on trigger
                insert_new_trigger(triggered_dataset, torch.zeros(args.trigger_length, device=DEVICE).long(), where_to_insert='attention_mask')
                
                # train loss to get train baseline
                compute_loss(models, triggered_dataset, args.batch_size, with_gradient=False, populate_baselines=True)
                
                # test loss to get train baseline
                models['clean_test'] = [model.to(DEVICE, non_blocking=True) for model in models['clean_test']]
                compute_loss(models, triggered_dataset, args.batch_size, with_gradient=False, train_or_test='test', populate_baselines=True)
                models['clean_test'] = [model.to(CPU, non_blocking=True) for model in models['clean_test']]

                # add back attention
                insert_new_trigger(triggered_dataset, torch.ones(args.trigger_length, device=DEVICE).long(), where_to_insert='attention_mask')
            insert_trigger_and_populate_baselines()
            
            def take_best_k_inputs(triggered_dataset, k=15):
                good_ixs = (triggered_dataset['train_eval_baseline_likelihoods'] < .5).nonzero().flatten()
                triggered_dataset = {k:v[good_ixs] for k,v in triggered_dataset.items() if not isinstance(v, tuple)}
                best_inputs = torch.topk(triggered_dataset['train_eval_answer_likelihoods'], min(k, len(triggered_dataset['input_ids'])))[1]
                triggered_dataset = {k:v[best_inputs] for k,v in triggered_dataset.items() if not isinstance(v, tuple)}
                triggered_dataset['q_trigger_mask_tuple'] = torch.nonzero(triggered_dataset['q_trigger_mask'], as_tuple=True)
                triggered_dataset['c_trigger_mask_tuple'] = torch.nonzero(triggered_dataset['c_trigger_mask'], as_tuple=True)
                return triggered_dataset
            triggered_dataset = take_best_k_inputs(triggered_dataset)

            def put_embeds_on_device(device=DEVICE):
                input_id_embeddings['eval']['avg'] = input_id_embeddings['eval']['avg'].to(device, non_blocking=True)
                input_id_embeddings['clean_train']['avg'] = input_id_embeddings['clean_train']['avg'].to(device, non_blocking=True)
            put_embeds_on_device()

            for i in range(args.num_random_tries):
                
                def initialize_trigger(trigger_init_fn):
                    # functions
                    def get_random_new_trigger():
                        return torch.tensor([randint(0,len(tokenizer.vocab)-1) for _ in range(args.trigger_length)]).to(DEVICE)
                    def get_pad_trigger():
                        return torch.tensor([tokenizer.pad_token_id]*args.trigger_length).to(DEVICE)
                    def pick_random_permutation_of_most_changed_embeds():
                        # return total_cand_pool[np.random.choice(len(total_cand_pool), size=args.trigger_length, replace=False)]
                        # return total_cand_pool[np.random.choice(len(total_cand_pool))]
                        
                        num_composed_words = random.randint(min(len(total_cand_pool['multi_token_words']), 1), len(total_cand_pool['multi_token_words']))
                        sampled_composed_words = random.sample(total_cand_pool['multi_token_words'], num_composed_words)

                        new_trigger_list = []
                        num_tokens_chosen = 0
                        for composed_word in sampled_composed_words:
                            if len(composed_word)+num_tokens_chosen < args.trigger_length:
                                new_trigger_list.append(composed_word)
                                num_tokens_chosen += len(composed_word)
                            else:
                                break
                        new_trigger_list += random.choices(total_cand_pool['single_token_words'], k=args.trigger_length - num_tokens_chosen)
                        new_arrangement = random.sample(range(len(new_trigger_list)), len(new_trigger_list))
                        new_trigger_list = [new_trigger_list[i] for i in new_arrangement]
                        unpacked_trigger_list = []
                        for token in new_trigger_list:
                            if isinstance(token, list):
                                unpacked_trigger_list += token
                            else:
                                unpacked_trigger_list.append(token)
                        return torch.stack(unpacked_trigger_list)
                            

                    # mapping
                    trigger_init_names_to_fn = {
                        'embed_ch': pick_random_permutation_of_most_changed_embeds,
                        'random': get_random_new_trigger, 
                        'pad': get_pad_trigger}
                    return trigger_init_names_to_fn[trigger_init_fn]()
                
                num_non_random_tries = args.num_random_tries//2
                if 'electra' in config['model_architecture']:
                    num_non_random_tries = 2*(args.num_random_tries//3)
                if i < num_non_random_tries:
                    new_trigger = initialize_trigger(args.trigger_init_fn)            
                else:
                    new_trigger = initialize_trigger('random')
                    # torch.uniform()len()
                insert_new_trigger(triggered_dataset, new_trigger)

                old_trigger, n_iter = torch.tensor([randint(0,20000) for _ in range(args.trigger_length)]).to(DEVICE), 0
                with autocast():         
                    while not torch.equal(old_trigger, new_trigger) and n_iter < args.max_iter:
                        # start_time = time.time()
                        
                        old_trigger = deepcopy(new_trigger)
                        old_loss = compute_loss(models, triggered_dataset, args.batch_size, with_gradient=True)

                        @torch.no_grad()
                        def find_best_k_candidates_for_each_trigger_token(old_trigger, num_candidates, tokenizer):    
                            '''
                            equation 2: (embedding_matrix - trigger embedding)T @ trigger_grad
                            '''
                            # put_embeds_on_device(device=DEVICE)

                            # [num_inputs, num_tokens_per_input, dimensionality]
                            embeds_shape = [len(triggered_dataset['input_ids']), -1, input_id_embeddings['eval']['avg'].shape[-1]]

                            def get_mean_trigger_grads(tokenizer, input_id_embeddings, eval_or_clean):
                                concat_grads = torch.cat(EXTRACTED_GRADS[eval_or_clean])
                                grads_list = []
                                if args.trigger_insertion_type in ['context', 'both']:
                                    mean_context_grads_over_inputs = concat_grads[triggered_dataset['c_trigger_mask']].view(embeds_shape).mean(dim=0)
                                    grads_list.append(mean_context_grads_over_inputs)
                                if args.trigger_insertion_type in ['question', 'both']:
                                    mean_question_grads_over_inputs = concat_grads[triggered_dataset['q_trigger_mask']].view(embeds_shape).mean(dim=0)
                                    grads_list.append(mean_question_grads_over_inputs)
                                return torch.stack(grads_list).mean(dim=0)                
                            eval_mean_trigger_grads = get_mean_trigger_grads(tokenizer, input_id_embeddings, 'eval')
                            clean_train_mean_trigger_grads = get_mean_trigger_grads(tokenizer, input_id_embeddings, 'clean_train')
                            
                            eval_grad_dot_embed_matrix  = torch.einsum("ij,kj->ik", (eval_mean_trigger_grads,  input_id_embeddings['eval']['avg']))
                            eval_grad_dot_embed_matrix[eval_grad_dot_embed_matrix != eval_grad_dot_embed_matrix] = 1e2
                            clean_grad_dot_embed_matrix = torch.einsum("ij,kj->ik", (clean_train_mean_trigger_grads, input_id_embeddings['clean_train']['avg']))
                            clean_grad_dot_embed_matrix[clean_grad_dot_embed_matrix != clean_grad_dot_embed_matrix] = 1e2

                            # fill nans
                            eval_grad_dot_embed_matrix[eval_grad_dot_embed_matrix != eval_grad_dot_embed_matrix] = 1e2
                            clean_grad_dot_embed_matrix[clean_grad_dot_embed_matrix != clean_grad_dot_embed_matrix] = 1e2

                            # weigh clean_train and eval dot products and get the smallest ones for each position
                            gradient_dot_embedding_matrix = eval_grad_dot_embed_matrix + LAMBDA*clean_grad_dot_embed_matrix 
                            BANNED_TOKEN_IDS = [tokenizer.pad_token_id, 
                                                tokenizer.cls_token_id, 
                                                tokenizer.unk_token_id, 
                                                tokenizer.sep_token_id, 
                                                tokenizer.mask_token_id]
                            for token_id in BANNED_TOKEN_IDS:
                                gradient_dot_embedding_matrix[:, token_id] = 1e2
                            _, best_k_ids = torch.topk(-gradient_dot_embedding_matrix, num_candidates, dim=1)

                            # put_embeds_on_device(device=CPU)
                            return best_k_ids
                        candidates = find_best_k_candidates_for_each_trigger_token(old_trigger, args.num_candidates, tokenizer)

                        variations = [old_trigger[np.random.choice(len(old_trigger), size=args.trigger_length, replace=False)]
                                                                                                for _ in range(args.num_variations)]
                        variations = torch.stack(variations).T
                        candidates = torch.cat([candidates, variations], dim=-1)

                        def clear_all_model_grads(models):
                            for model_type, model_list in models.items():
                                for model in model_list:
                                    optimizer = optim.Adam(model.parameters())
                                    optimizer.zero_grad(set_to_none=True)
                            for model_type in EXTRACTED_GRADS.keys():
                                EXTRACTED_GRADS[model_type] = []
                        clear_all_model_grads(models)

                        @torch.no_grad()
                        def evaluate_and_pick_best_candidate(candidates, beam_size):
                            @torch.no_grad()
                            def evaluate_candidate_tokens_for_pos(candidates, triggered_dataset, top_candidate, pos):
                                @torch.no_grad()
                                def evaluate_loss_with_temp_trigger(triggered_dataset, temp_trigger):
                                    insert_new_trigger(triggered_dataset, temp_trigger)
                                    loss = compute_loss(models, triggered_dataset, args.batch_size*8, with_gradient=False)
                                    return [loss['trigger_inversion_loss'], loss['clean_loss'], loss['eval_loss'], loss['clean_asr'], loss['eval_asr'], deepcopy(temp_trigger)]

                                top_cand = deepcopy(top_candidate[-1])
                                loss_per_candidate_trigger = [deepcopy(top_candidate)]
                                visited_triggers = set(top_cand)
                                
                                for candidate_token in candidates[pos]:
                                    temp_trigger = top_cand
                                    temp_trigger[pos] = candidate_token
                                    if temp_trigger in visited_triggers:
                                            continue
                                    temp_result = evaluate_loss_with_temp_trigger(triggered_dataset, temp_trigger)
                                    loss_per_candidate_trigger.append(temp_result)
                                    visited_triggers.add(temp_result[-1])

                                return loss_per_candidate_trigger
                            
                            top_candidate = [old_loss['trigger_inversion_loss'], old_loss['clean_loss'], old_loss['eval_loss'], old_loss['clean_asr'], old_loss['eval_asr'], old_trigger]
                            loss_per_candidate_trigger = evaluate_candidate_tokens_for_pos(candidates, triggered_dataset, top_candidate, pos=0)
                            top_candidates = heapq.nsmallest(beam_size, loss_per_candidate_trigger, key=itemgetter(0))
                                                            
                            for idx in range(1, len(old_trigger)):
                                loss_per_candidate_trigger = []
                                for top_candidate in top_candidates:
                                    loss_per_candidate_trigger.extend(evaluate_candidate_tokens_for_pos(candidates, triggered_dataset, top_candidate, pos=idx))
                                top_candidates = heapq.nsmallest(beam_size, loss_per_candidate_trigger, key=itemgetter(0))
                            trigger_inversion_loss, clean_loss, eval_loss, clean_triggered, eval_triggered, new_trigger = min(top_candidates, key=itemgetter(0))
                            new_loss = {'trigger_inversion_loss': trigger_inversion_loss,
                                        'eval_loss': eval_loss,
                                        'clean_loss': clean_loss,
                                        'clean_asr': clean_triggered,
                                        'eval_asr': eval_triggered}
                            return new_loss, new_trigger   
                        new_loss, new_trigger = evaluate_and_pick_best_candidate(candidates, args.beam_size)
                        insert_new_trigger(triggered_dataset, new_trigger)

                        n_iter += 1
                        def print_results_of_trigger_inversion_iterate(n_iter):
                            print(f"Iteration: {n_iter}")
                            print(f'Example: {tokenizer.decode(triggered_dataset["input_ids"][-1])}')
                            table = Texttable()
                            table.add_rows([['','trigger', 'trigger_inversion_loss', 'eval/clean asr'],
                                            ['old', tokenizer.decode(old_trigger), 
                                            f"{round(old_loss['trigger_inversion_loss'].item(), 2)} = {round(old_loss['eval_loss'].item(), 2)} + {LAMBDA}*{round(old_loss['clean_loss'].item(), 2)}",
                                            f"{round(old_loss['eval_asr'].item()*100, 1)} / {round(old_loss['clean_asr'].item()*100, 1)}"],
                                            ['new', tokenizer.decode(new_trigger), 
                                            f"{round(new_loss['trigger_inversion_loss'].item(), 2)} = {round(new_loss['eval_loss'].item(), 2)} + {LAMBDA}*{round(new_loss['clean_loss'].item(), 2)}",
                                            f"{round(new_loss['eval_asr'].item()*100, 1)} / {round(new_loss['clean_asr'].item()*100, 1)}"]])
                            print(table.draw())
                        if args.is_submission == False:
                            print_results_of_trigger_inversion_iterate(n_iter)
                        
                        del old_loss, candidates

                    models['clean_test'] = [model.to(DEVICE, non_blocking=True) for model in models['clean_test']]
                    new_test_loss = compute_loss(models, triggered_dataset, args.batch_size, with_gradient=False, train_or_test='test')
                    models['clean_test'] = [model.to(CPU, non_blocking=True) for model in models['clean_test']]
                if best_test_loss is None or best_test_loss['trigger_inversion_loss'] > new_test_loss['trigger_inversion_loss']:
                    best_trigger, best_train_loss, best_test_loss = deepcopy(new_trigger), deepcopy(new_loss), deepcopy(new_test_loss)
                
                if best_test_loss['trigger_inversion_loss'] < 0.01:
                    # found_trigger_flag = True
                    break
                torch.cuda.empty_cache()
            
            def add_results_to_df(df):
                end_time = time.time()
                temp_df = pd.DataFrame.from_dict({
                        # trigger
                            'decoded_trigger':              tokenizer.decode(best_trigger), 
                            'trigger_token_ids':            best_trigger.detach().cpu().numpy(),
                        # train
                            'train_clean_asr':              round(best_train_loss['clean_asr'].item(), 5),
                            'train_eval_asr':               round(best_train_loss['eval_asr'].item(), 5),
                            'train_trigger_inversion_loss': round(best_train_loss['trigger_inversion_loss'].item(), 5), 
                            'train_eval_loss':              round(best_train_loss['eval_loss'].item(), 5),
                            'train_clean_loss':             round(best_train_loss['clean_loss'].item(), 5),
                        # test
                            'test_clean_asr':               round(best_test_loss['clean_asr'].item(), 5),
                            'test_eval_asr':                round(best_test_loss['eval_asr'].item(), 5),
                            'test_trigger_inversion_loss':  round(best_test_loss['trigger_inversion_loss'].item(), 5), 
                            'test_eval_loss':               round(best_test_loss['eval_loss'].item(), 5),
                            'test_clean_loss':              round(best_test_loss['clean_loss'].item(), 5),
                            'smallest_values':              total_cand_pool['smallest_values'].detach().cpu().numpy(),
                            'total_similarity':             torch.tensor(list(total_cand_pool['total_similarity'])).detach().cpu().numpy(),
                        # time
                            'total_time':                   round(end_time - start_time, 1),
                        # config
                            'behavior':                     args.trigger_behavior,
                            'insertion':                    args.trigger_insertion_type,
                            'num_inputs':                   len(triggered_dataset['input_ids'])
                    }, orient='index')
                temp_df = temp_df.T
                return df.append(temp_df)
            df = add_results_to_df(df)


    if not args.is_submission:
        def save_results(parent_folder='results'):
            '''
            saves results to the results folder
            '''
            def check_if_folder_exists(folder):
                if not os.path.isdir(folder):
                    os.mkdir(folder)
            
            check_if_folder_exists(parent_folder)
            
            folder = f'{parent_folder}/nov16'+\
                        f'_lam_{args.lmbda}'+\
                        f'_trinv_{args.trigger_inversion_method}'+\
                        f'_ncand_{args.num_candidates}'+\
                        f'_trlen_{args.trigger_length}'+\
                        f'_nrand_{args.num_random_tries}'+\
                        f'_nbeam_{args.beam_size}'+\
                        f'_mordt_{args.more_clean_data}'+\
                        f'_niter_{args.max_iter}'+\
                        f'_initf_{args.trigger_init_fn}'+\
                        f'_agg_{args.likelihood_agg}'
            check_if_folder_exists(folder)
            
            df.to_csv(os.path.join(folder, f'{args.model_num}.csv'))
        save_results()
    

    if args.is_submission:
        df['behavior_insertion'] = df['behavior'] + '_' + df['insertion']
        normalization_df = pd.read_csv(f'{"/" if args.is_submission else ""}normalization_df', index_col='behavior_insertion')
        df['normalized_test_trigger_inversion_loss'] = \
            df.apply(lambda x: (x.test_trigger_inversion_loss-normalization_df.loc[x.behavior_insertion]['mean']).item()/normalization_df.loc[x.behavior_insertion]['std'].item(), axis=1)
        X = df.sort_values('normalized_test_trigger_inversion_loss').iloc[[0]][['normalized_test_trigger_inversion_loss', 'test_clean_asr', 'test_eval_asr']]
        
        from joblib import load
        clf = load(f'{"/" if args.is_submission else ""}classifier.joblib')
        trojan_probability = clf.predict_proba(X)[0][1]

        with open(args.result_filepath, 'w') as fh:
            fh.write("{}".format(trojan_probability))


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Trojan Detector With Trigger Inversion for Question & Answering Tasks.')

    def add_all_args(parser):
        # general_args: remember to switch is_submission to 1 when making a container
        parser.add_argument('--is_submission',      dest='is_submission',   action='store_true',  help='Flag to determine if this is a submission to the NIST server',  )
        parser.add_argument('--calculate_alpha',    dest='calculate_alpha', action='store_true',  help='Flag to determine if we want to save the alphas of the evaluation model',  )
        parser.add_argument('--more_clean_data',    dest='more_clean_data', action='store_true',  help='Flag to determine if we want to grab clean examples from the clean models',  )
        parser.add_argument('--model_num',          default=12,             type=int,             help="model number - only used if it's not a submission")                    
        parser.add_argument('--batch_size',         default=32,             type=int,             help='What batch size')
        parser.add_argument('--max_test_models',    default=7,              type=int,             help='How many test models to use', choices=range(2, 7))

        # trigger_inversion_args
        parser.add_argument('--trigger_inversion_method',     default='discrete',  type=str,   help='Which trigger inversion method do we use', choices=['discrete', 'relaxed'])
        parser.add_argument('--trigger_behavior',             default='cls',      type=str,   help='Where does the trigger point to?', choices=['self', 'cls'])
        parser.add_argument('--likelihood_agg',               default='max',       type=str,   help='How do we aggregate the likelihoods of the answers', choices=['max', 'sum'])
        parser.add_argument('--trigger_insertion_type',       default='question',      type=str,   help='Where is the trigger inserted', choices=['context', 'question', 'both'])
        parser.add_argument('--num_random_tries',             default=15,          type=int,   help='How many random starts do we try')
        parser.add_argument('--trigger_length',               default=7,           type=int,   help='How long do we want the trigger to be')
        parser.add_argument('--lmbda',                        default=2.,          type=float, help='Weight on the clean loss')
        parser.add_argument('--temperature',                  default=1.,          type=float, help='Temperature parameter to divide logits by')

        # discrete
        parser.add_argument('--trigger_init_fn', default='embed_ch', type=str, help='How do we initialize our trigger', choices=['embed_ch', 'random', 'pad'])
        parser.add_argument('--max_iter',        default=20,         type=int, help='Max num of iterations', choices=range(0,50))
        parser.add_argument('--num_variations',  default=1,         type=int, help='How many variations to evaluate for each position during the discrete trigger inversion')
        parser.add_argument('--num_candidates',  default=1,          type=int, help='How many candidates do we want to evaluate in each position during the discrete trigger inversion')
        parser.add_argument('--beam_size',       default=2,          type=int, help='How big do we want the beam size during the discrete trigger inversion')
        
        # continuous
        parser.add_argument('--beta',           default=.01,      type=float, help='Weight on the sparsity loss')
        parser.add_argument('--rtol',           default=1e-3,     type=float, help="How different are the w's before we stop")
        
        # do not touch
        parser.add_argument('--model_filepath',     type=str, help='Filepath to the pytorch model file to be evaluated.', default='./model/model.pt')
        parser.add_argument('--tokenizer_filepath', type=str, help='Filepath to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default='./tokenizers/google-electra-small-discriminator.pt')
        parser.add_argument('--result_filepath',    type=str, help='Filepath to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
        parser.add_argument('--scratch_dirpath',    type=str, help='Filepath to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
        parser.add_argument('--examples_dirpath',   type=str, help='Filepath to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.', default='./model/example_data')

        return parser
    parser = add_all_args(parser)
    parser.set_defaults(is_submission=True, more_clean_data=True, calculate_alpha=False)
    args = parser.parse_args()
    
    def modify_args(args):
        global CLEAN_TRAIN_MODELS_FILEPATH, CLEAN_TEST_MODELS_FILEPATH
        CLEAN_TRAIN_MODELS_FILEPATH = SUBMISSION_CLEAN_TRAIN_MODELS_FILEPATH
        CLEAN_TEST_MODELS_FILEPATH = SUBMISSION_CLEAN_TEST_MODELS_FILEPATH
        if not args.is_submission:
            metadata = pd.read_csv(join(SANDBOX_TRAINING_FILEPATH, 'METADATA.csv'))

            id_str = str(100000000 + args.model_num)[1:]
            model_id = 'id-'+id_str

            args.model_filepath = join(SANDBOX_TRAINING_FILEPATH, 'models', model_id, 'model.pt')
            args.examples_dirpath = join(SANDBOX_TRAINING_FILEPATH, 'models', model_id, 'example_data')
            CLEAN_TRAIN_MODELS_FILEPATH = SANDBOX_CLEAN_TRAIN_MODELS_FILEPATH
            CLEAN_TEST_MODELS_FILEPATH = SANDBOX_CLEAN_TEST_MODELS_FILEPATH
        global LAMBDA
        LAMBDA = args.lmbda
        args.eval_model_filepath = args.model_filepath
        return args
    args = modify_args(args)

    trojan_detector(args)