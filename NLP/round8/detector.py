'''
NOTES
- Assume that the code will not pass answer position information
- Loss function will either add up or max logits
- Understand how to combine loss from start and end
    - Averaging
- The trigger should be introduced in both the question and the answer
    - Do we need to be careful when introducing the trigger to avoid adding it to examples that do not contain the context?
QUESTIONS
- Discover what is the point of token_type_ids in distilbert
    encoded_dict['token_type_ids']
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
- Can we just add to the current context without overflowing? Let's assume yes.
'''
import argparse
import os
from os.path import join

import datasets
import pandas as pd
import torch
import transformers
import json
from copy import deepcopy
from filepaths import TRAINING_FILEPATH, CLEAN_TRAIN_MODELS_FILEPATH, CLEAN_TEST_MODELS_FILEPATH

import warnings

import utils_qa

warnings.filterwarnings("ignore")

''' CONSTANTS '''
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXTRACTED_GRADS = {'eval':[], 'clean':[]}

    
def insert_new_trigger(triggered_dataset, new_trigger):
    '''
    Replaces the current trigger in the input_ids of a triggered_dataset for a new_trigger
    Returns an updated triggered_dataset
    '''
    new_trigger = torch.tensor(new_trigger)

    def insert_new_trigger_helper(dataset_sample):
        dataset_sample['input_ids'][dataset_sample['q_trigger_mask']] = new_trigger
        dataset_sample['input_ids'][dataset_sample['c_trigger_mask']] = new_trigger
        return dataset_sample

    triggered_dataset = triggered_dataset.map(
        insert_new_trigger_helper,
        batched=False,
        num_proc=1,
        keep_in_memory=True)
    
    return triggered_dataset


def get_fwd_var_list(model):
    var_list = ['input_ids', 'attention_mask']
    if ('distilbert' not in model.name_or_path) and ('bart' not in model.name_or_path):
            var_list += ['token_type_ids']
    return var_list


def compute_loss(models, dataloader, with_gradient=False, train_or_test='train'):
    '''
    Computes the trigger inversion loss over all examples in the dataloader
    '''
    # TODO: Check the shape of the gradient when we have a batch!
    #   If we do not have separate gradients per input we have to use batch_size=1 
    assert train_or_test in ['train', 'test']
    var_list = get_fwd_var_list(models['eval'][0])
    losses = []
    for _, batch in enumerate(dataloader):  
        all_logits = {'eval_start':[], 'clean_start': [], 'eval_end': [], 'clean_end': []}

        def get_batch_loss():
            def add_logits(clean_or_eval, output):
                all_logits[f'{clean_or_eval}_start'].append(output['start_logits'])
                all_logits[f'{clean_or_eval}_end'].append(output['end_logits'])
            add_logits('eval', models['eval'][0](**{v:batch[v].to(DEVICE) for v in var_list}))
            for clean_model in models[f'clean_{train_or_test}']:
                add_logits('clean', clean_model(**{v:batch[v].to(DEVICE) for v in var_list}))
            
            return trigger_inversion_loss_fn(batch, all_logits)
        
        if with_gradient == False:
            with torch.no_grad():
                batch_loss = get_batch_loss()
        else:
            batch_loss = get_batch_loss()
            batch_loss.backward()
            def clear_gradients():
                for _, model_list in models.items():
                    for model in model_list:
                        model.zero_grad()
            clear_gradients()
        losses.append(batch_loss.detach())
    return torch.stack(losses).mean()


def trigger_inversion_loss_fn(batch, all_logits):
    '''
    Returns the trigger inversion loss, which is:
        clean_loss + lambda*evaluation_loss
    
    The clean_loss is the average of the start and end loss:
        (avg_clean_start_loss + avg_clean_end_loss)/2
    
    We use the clean_loss_by_pos function to calculate either the start or end loss per model
        The probability that the answer is in the cls_token or in the trigger should be small
        Note that these probabilities can be constructed with softmax over the set of valid tokens
        We also use the 'net' likelihood by subtracting the 'baseline' likelihood, derived from clean inputs
        Finally, we take the max over the net likelihood of cls or trigger and compute the log-likelihood
    '''
    softmax = torch.nn.Softmax()
    trigger_length = batch['c_trigger_mask'].sum(-1)[0]

    def clean_loss():
        def avg_start_and_end_clean_loss():

            def clean_loss_by_pos(pos, logits):
                assert pos in ['start', 'end']
                def get_pos_ix(pos):
                    pos_ix = 0
                    if pos == 'end':
                        pos_ix = 1
                    return pos_ix
                pos_ix = get_pos_ix(pos)

                valid_outputs_mask = (batch['context_mask'] | batch['cls_mask']).bool()
                # TODO: Optimize this cuda
                valid_likelihoods = softmax(logits - (~valid_outputs_mask.to(DEVICE))*1e10)
                
                cls_likelihood = valid_likelihoods[batch['cls_mask'].bool()]
                net_cls_likelyhood = torch.max(cls_likelihood - batch['clean_cls_likelihoods'][:, pos_ix].to(DEVICE), 
                                                torch.zeros_like(cls_likelihood, device=DEVICE))
                            
                trigger_likelihood = valid_likelihoods[batch['c_trigger_mask']].reshape([-1, trigger_length]).sum(dim=1)
                base_trigger_likelihood = (trigger_length/valid_outputs_mask).sum(dim=-1).median()
                net_trigger_likelyhood = torch.max(trigger_likelihood - base_trigger_likelihood, 
                                                    torch.zeros_like(trigger_likelihood, device=DEVICE))          
                
                return -torch.log(1-torch.max(net_cls_likelyhood, net_trigger_likelyhood)).mean()
            
            clean_losses = {'start':[], 'end': []}
            for start_logits, end_logits in zip(all_logits['clean_start'], all_logits['clean_end']):
                clean_losses['start'].append(clean_loss_by_pos('start', start_logits))
                clean_losses['end'].append(clean_loss_by_pos('end', end_logits))
            average_clean_start_loss = torch.stack(clean_losses['start']).mean(dim=0)
            average_clean_end_loss = torch.stack(clean_losses['end']).mean(dim=0)
            return average_clean_start_loss, average_clean_end_loss

        avg_clean_start_loss, avg_clean_end_loss = avg_start_and_end_clean_loss()
        return (avg_clean_start_loss + avg_clean_end_loss)/2

    def evaluation_loss():
        def eval_loss_pos(logits):
            valid_outputs_mask = (batch['context_mask'] | batch['cls_mask']).bool()
            # TODO: Optimize this cuda
            valid_likelihoods = softmax(logits - (~valid_outputs_mask.to(DEVICE))*1e10)
            
            cls_likelihood = valid_likelihoods[batch['cls_mask'].bool()]
            trigger_likelihood = valid_likelihoods[batch['c_trigger_mask']].reshape([-1, trigger_length]).sum(dim=1)

            return -torch.log(torch.max(cls_likelihood, trigger_likelihood)).mean()

        return (eval_loss_pos(all_logits['eval_start'][0]) + eval_loss_pos(all_logits['eval_end'][0]))/2

    return clean_loss() + LAMBDA*evaluation_loss()


def trojan_detector(args):
    '''
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
    '''
    print(args)

    def load_config(model_filepath):
        model_dirpath, _ = os.path.split(model_filepath)
        with open(os.path.join(model_dirpath, 'config.json')) as json_file:
            config = json.load(json_file)
        print('Source dataset name = "{}"'.format(config['source_dataset']))
        if 'data_filepath' in config.keys():
            print('Source dataset filepath = "{}"'.format(config['data_filepath']))
        return config
    config = load_config(args.eval_model_filepath)

    def load_all_models(eval_model_filepath, config):
        def load_model(model_filepath):
            classification_model = torch.load(model_filepath, map_location=DEVICE)
            classification_model.eval()
            return classification_model
        
        classification_model = load_model(eval_model_filepath)

        def get_clean_model_filepaths(config, is_testing=False):
            key = f"{config['source_dataset'].lower()}_{config['model_architecture'].split('/')[-1]}"
            model_name = config['output_filepath'].split('/')[-1]
            base_path = CLEAN_TRAIN_MODELS_FILEPATH
            if is_testing:
                base_path = CLEAN_TEST_MODELS_FILEPATH
            model_folders = [f for f in os.listdir(base_path) \
                                if (key in f and model_name not in f)]
            clean_classification_model_paths = \
                [join(base_path, model_folder, 'model.pt') for model_folder in model_folders]       
            return clean_classification_model_paths

        def load_clean_models(clean_model_filepath):
            clean_models = []
            for f in clean_model_filepath:
                clean_models.append(load_model(f))
            return clean_models
        
        clean_models_filepath = get_clean_model_filepaths(config, is_testing=False)
        clean_models_train = load_clean_models(clean_models_filepath)

        clean_testing_model_filepath = get_clean_model_filepaths(config, is_testing=True)
        clean_models_test = load_clean_models(clean_testing_model_filepath)

        return {'eval': [classification_model],
                'clean_train': clean_models_train,
                'clean_test': clean_models_test}
    models = load_all_models(args.eval_model_filepath, config)   
    
    def add_hooks_to_all_models(models):
        def add_hooks(model, is_clean):
            
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
                    EXTRACTED_GRADS['clean'].append(grad_out[0]) 
                module.register_backward_hook(extract_clean_grad_hook)
            else:
                def extract_grad_hook(module, grad_in, grad_out):
                    EXTRACTED_GRADS['eval'].append(grad_out[0])  

                module.register_backward_hook(extract_grad_hook)

        add_hooks(models['eval'][0], is_clean=False)

        for clean_model in models['clean_train']:
            add_hooks(clean_model, is_clean=True)
    add_hooks_to_all_models(models)

    def load_tokenizer(is_submission, tokenizer_filepath, config):
        if is_submission:
            tokenizer = torch.load(tokenizer_filepath)
        else:
            model_architecture = config['model_architecture']
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_architecture, use_fast=True)
        return tokenizer
    tokenizer = load_tokenizer(args.is_submission, args.tokenizer_filepath, config)

    def load_dataset(examples_dirpath, scratch_dirpath):
        # clean example inference
        fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
        fns.sort()
        examples_filepath = fns[0]

        # Load the examples
        # TODO The cache_dir is required for the test server since /home/trojai is not writable and the default cache locations is ~/.cache
        dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))
        return dataset
    dataset = load_dataset(args.examples_dirpath, args.scratch_dirpath)
    
    @torch.no_grad()
    def tokenize_for_qa(tokenizer, dataset, models):

        question_column_name, context_column_name, answer_column_name  = "question", "context", "answers"
        
        # Padding side determines if we do (question|context) or (context|question).
        pad_on_right = tokenizer.padding_side == "right"
        max_seq_length = min(tokenizer.model_max_length, 384)
        
        if 'mobilebert' in tokenizer.name_or_path:
            max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
        
        # Training preprocessing
        def prepare_train_features(examples):
            # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
            # in one example possible giving several features when a context is long, each of those features having a
            # context that overlaps a bit the context of the previous feature.
            
            pad_to_max_length = True
            doc_stride = 128
            tokenized_examples = tokenizer(
                examples[question_column_name if pad_on_right else context_column_name],
                examples[context_column_name if pad_on_right else question_column_name],
                truncation="only_second" if pad_on_right else "only_first",
                max_length=max_seq_length,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length" if pad_to_max_length else False,
                return_token_type_ids=True)  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created
            
            # populate the cls start and end likelyhoods
            
            clean_outputs = {'start_logits':[], 'end_logits': []}
            var_list = get_fwd_var_list(models['eval'][0])
            for clean_model in models['clean_train']:
                clean_output = clean_model(**{v:torch.tensor(tokenized_examples[v]).to(DEVICE) for v in var_list})
                clean_outputs['start_logits'].append(clean_output['start_logits'])
                clean_outputs['end_logits'].append(clean_output['end_logits'])
            
            # initialize lists
            var_list = ['question_start_and_end', 'context_start_and_end', 
                        'clean_cls_likelihoods', 'answer_start_and_end']
            for var_name in var_list:
                tokenized_examples[var_name] = []
            
            sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
            # Let's label those examples!
            for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
                # We will label impossible answers with the index of the CLS token.
                input_ids = tokenized_examples["input_ids"][i]
                cls_ix = input_ids.index(tokenizer.cls_token_id)
                
                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = tokenized_examples.sequence_ids(i)
                
                context_ix = 1 if pad_on_right else 0
                question_ix = 0 if pad_on_right else 1
                
                # One example can give several spans, this is the index of the example containing this span of text.
                sample_ix = sample_mapping[i]
                answers = examples[answer_column_name][sample_ix]

                def get_token_index(sequence_ids, input_ids, index, is_end):
                    token_ix = 0
                    if is_end: 
                        token_ix = len(input_ids) - 1
                    
                    add_num = 1
                    if is_end:
                        add_num = -1

                    while sequence_ids[token_ix] != index:
                        token_ix += add_num
                    return token_ix

                # populate question_start_and_end
                token_question_start_ix = get_token_index(sequence_ids, input_ids, index=question_ix, is_end=False)
                token_question_end_ix   = get_token_index(sequence_ids, input_ids, index=question_ix, is_end=True)

                tokenized_examples["question_start_and_end"].append([token_question_start_ix, token_question_end_ix])

                # populate context_start_and_end
                token_context_start_ix = get_token_index(sequence_ids, input_ids, index=context_ix, is_end=False)
                token_context_end_ix   = get_token_index(sequence_ids, input_ids, index=context_ix, is_end=True)

                tokenized_examples["context_start_and_end"].append([token_context_start_ix, token_context_end_ix])

                # TODO: Test if context_start_and_end is correct

                def set_answer_start_and_end_to_ixs(first_ix, second_ix):
                    tokenized_examples["answer_start_and_end"].append([first_ix, second_ix])

                # If no answers are given, set the cls_index as answer.
                if len(answers["answer_start"]) == 0:
                    set_answer_start_and_end_to_ixs(cls_ix, cls_ix)
                else:
                    # Start/end character index of the answer in the text.
                    start_char = answers["answer_start"][0]
                    end_char = start_char + len(answers["text"][0])
                    
                    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                    if (start_char < offsets[token_context_start_ix][0] or offsets[token_context_end_ix][1] < end_char):
                        set_answer_start_and_end_to_ixs(cls_ix, cls_ix)
                    else:
                        token_answer_start_ix = token_context_start_ix
                        token_answer_end_ix = token_context_end_ix
                        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                        # Note: we could go after the last offset if the answer is the last word (edge case).
                        while token_answer_start_ix < len(offsets) and offsets[token_answer_start_ix][0] <= start_char:
                            token_answer_start_ix += 1
                        while offsets[token_answer_end_ix][1] >= end_char:
                            token_answer_end_ix -= 1
                        set_answer_start_and_end_to_ixs(token_answer_start_ix-1, token_answer_end_ix+1)
                
                # This is for the evaluation side of the processing
                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.
                tokenized_examples["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_ix else None)
                    for k, o in enumerate(tokenized_examples["offset_mapping"][i])
                ]

                # check that cls_index is not within context
                assert cls_ix < token_context_start_ix or token_context_end_ix < cls_ix
                relevant_logits_ix_list = [cls_ix] + list(range(token_context_start_ix, token_context_end_ix+1))
                # populate the mean clean cls likelyhood for each position
                softmax = torch.nn.Softmax()

                clean_cls_likelyhoods = []
                for pos in ['start', 'end']:
                    cls_likelihood_list = []
                    for logits in clean_outputs[f'{pos}_logits']:
                        cls_likelihood_list.append(softmax(logits[i][relevant_logits_ix_list])[0])
                    clean_cls_likelyhoods.append(torch.stack(cls_likelihood_list).mean(0))
                tokenized_examples['clean_cls_likelihoods'].append(clean_cls_likelyhoods)

            return tokenized_examples
        
        tokenized_dataset = dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=1,
            remove_columns=dataset.column_names,
            keep_in_memory=True)
        # tokenized_dataset.set_format('pt', columns=tokenized_dataset.column_names)
        tokenized_dataset = tokenized_dataset.remove_columns(['offset_mapping'])
        assert len(tokenized_dataset) > 0

        return tokenized_dataset
    tokenized_dataset = tokenize_for_qa(tokenizer, dataset, models)
    
    def select_non_cls_examples():
        answer_starts = torch.tensor(tokenized_dataset['answer_start_and_end'])[:, 0]
        non_cls_answer_indices = (~torch.eq(answer_starts, tokenizer.cls_token_id)).nonzero().flatten()
        return tokenized_dataset.select(non_cls_answer_indices)
    tokenized_dataset = select_non_cls_examples()


    def initialize_dummy_trigger(tokenized_dataset, tokenizer, trigger_length, trigger_insertion_locations):

        is_context_first = tokenizer.padding_side != 'right'
        
        def initialize_dummy_trigger_helper(dataset_instance):

            input_id, att_mask, token_type, q_pos, c_pos = [deepcopy(torch.tensor(dataset_instance[x])) for x in \
                ['input_ids', 'attention_mask', 'token_type_ids', 'question_start_and_end', 'context_start_and_end']]
            
            for var_name in ['input_ids', 'attention_mask', 'token_type_ids', 'q_trigger_mask', 'c_trigger_mask']:
                dataset_instance[var_name] = None

            def get_ix(insertion_location, start_end_ix, is_second_trigger=False):            
                offset = 0
                if is_second_trigger:
                    offset += 1
                if insertion_location == 'start':
                    return start_end_ix[0]
                elif insertion_location == 'end':
                    return start_end_ix[1]
                else:
                    print('please enter either "start" or "end" as an insertion_location')
            
            q_idx = get_ix(trigger_insertion_locations[0], q_pos)
            c_idx = get_ix(trigger_insertion_locations[1], c_pos)

            q_trigger_id, c_trigger_id = -1, -2
            q_trigger = torch.tensor([q_trigger_id]*trigger_length).long()
            c_trigger = torch.tensor([c_trigger_id]*trigger_length).long()

            first_idx, second_idx = q_idx, c_idx+1
            first_trigger, second_trigger = q_trigger, c_trigger
            if is_context_first:
                first_idx, second_idx = c_idx, q_idx+1
                first_trigger, second_trigger = c_trigger, q_trigger

            def insert_tensors_in_var(var, first_tensor, second_tensor=None):
                if second_tensor is None:
                    second_tensor = first_tensor
                var_copy = deepcopy(var)
                new_var = torch.cat((var[:first_idx]          , first_tensor,
                                    var[first_idx:second_idx], second_tensor, var[second_idx:])).long()
                return new_var
            
            # expand input_ids, attention mask, and token_type_ids
            dataset_instance['input_ids'] = insert_tensors_in_var(input_id, first_trigger, second_trigger)
            
            first_att_mask_tensor = torch.zeros(trigger_length) + att_mask[first_idx].item()
            second_att_mask_tensor = torch.zeros(trigger_length) + att_mask[second_idx].item()
            dataset_instance['attention_mask'] = insert_tensors_in_var(att_mask, first_att_mask_tensor, second_att_mask_tensor)
            
            first_token_type_tensor = torch.zeros(trigger_length) + token_type[first_idx].item()
            second_token_type_tensor = torch.zeros(trigger_length) + token_type[second_idx].item()
            dataset_instance['token_type_ids'] = insert_tensors_in_var(token_type, first_token_type_tensor, second_token_type_tensor)

            # make question and context trigger mask
            dataset_instance['q_trigger_mask'] = torch.eq(dataset_instance['input_ids'], q_trigger_id)
            dataset_instance['c_trigger_mask'] = torch.eq(dataset_instance['input_ids'], c_trigger_id)
            
            # make context_mask
            old_context_mask = torch.zeros_like(input_id)
            old_context_mask[dataset_instance['context_start_and_end'][0]:
                            dataset_instance['context_start_and_end'][1]+1] = 1
            dataset_instance['context_mask'] = insert_tensors_in_var(old_context_mask, torch.ones(trigger_length))

            # make cls_mask
            input_ids = dataset_instance["input_ids"]
            cls_ix = input_ids.tolist().index(tokenizer.cls_token_id)

            cls_mask = torch.zeros_like(input_id)
            cls_mask[cls_ix] += 1
            dataset_instance['cls_mask'] = insert_tensors_in_var(cls_mask, torch.zeros(trigger_length))

            return dataset_instance
        
        triggered_dataset = tokenized_dataset.map(
            initialize_dummy_trigger_helper,
            batched=False,
            num_proc=2,
            keep_in_memory=True)

        triggered_dataset = triggered_dataset.remove_columns([f'{v}_start_and_end' for v in ['question', 'context', 'answer']])
        triggered_dataset.set_format('pt', columns=triggered_dataset.column_names)

        return triggered_dataset
    triggered_dataset = initialize_dummy_trigger(tokenized_dataset, tokenizer, \
                                                args.trigger_length, args.trigger_insertion_locations)
    
    # insert new trigger
    new_trigger = [100] * args.trigger_length
    triggered_dataset = insert_new_trigger(triggered_dataset, new_trigger)
    dataloader = torch.utils.data.DataLoader(triggered_dataset, batch_size=10, shuffle=False)

    # make a dataloader including the embeddings 
    def map_to_token_embeds(triggered_dataset, models):
        # Map tokens to embeddings, adds three new tensors per input sample: 
        # 'token_embeds_eval', 'token_embeds_clean_train', and 'token_embeds_clean_test'. 
        # New tensors are of shape (num_models, max_input, embedding_size).

        def map_to_token_embeds_closure(dataset_sample):

            def embeds_fwd(model): 
                # Pass inputs through the model's embedding layers 
                var_list = ['input_ids', 'token_type_ids']
                singleton_batch = {v:dataset_sample[v].unsqueeze(0).to(DEVICE) for v in var_list}
                return next(model.children()).embeddings(**singleton_batch)

            # Embeddings for evaluation model 
            embeds_eval = embeds_fwd(models['eval'][0])

            # Embeddings for training models
            embeds_clean_train = [embeds_fwd(m) for m in models['clean_train']]
            embeds_clean_train = torch.stack(embeds_clean_train, dim=1).squeeze(0) 

            # Embeddings for test models 
            embeds_clean_test = [embeds_fwd(m) for m in models['clean_test']]
            embeds_clean_test = torch.stack(embeds_clean_test, dim=1).squeeze(0) 

            dataset_sample['token_embeds_eval'] = embeds_eval
            dataset_sample['token_embeds_clean_train'] = embeds_clean_train
            dataset_sample['token_embeds_clean_test'] = embeds_clean_test

            return dataset_sample 

        # BUG | Currently `map` is converting new tensors to lists of lists of tensors 
        token_embeds_dataset = triggered_dataset.map(
            map_to_token_embeds_closure,
            batched=False,
            num_proc=1,
            keep_in_memory=True)

        return token_embeds_dataset
    embeds_dataset = map_to_token_embeds(triggered_dataset, models)
    embeds_dataloader = torch.utils.data.DataLoader(embeds_dataset, batch_size=10, shuffle=False)

    # clean up
    del dataset, tokenized_dataset

    var_list = get_fwd_var_list(models['eval'][0])

    # trigger inversion loop
    old_trigger = None
    while (old_trigger != new_trigger):
        old_trigger = deepcopy(new_trigger)
        old_loss = compute_loss(models, dataloader, with_gradient=True)

        def get_candidates():
            return NotImplementedError
        candidates = get_candidates(args.num_candidates)

        def evaluate_and_pick_best_candidate():
            return NotImplementedError
        new_trigger, new_loss = evaluate_and_pick_best_candidate(candidates)

        def print_results_of_trigger_inversion_iterate():
            print(f'old trigger: {tokenizer.decode(old_trigger)} \t old loss: {round(old_loss.detach().item(),3)}')
            print(f'new trigger: {tokenizer.decode(new_trigger)} \t new loss: {round(new_loss.detach().item(),3)}')
        print_results_of_trigger_inversion_iterate()
        
        triggered_dataset = insert_new_trigger(triggered_dataset, new_trigger)
        dataloader = torch.utils.data.DataLoader(triggered_dataset, batch_size=10, shuffle=False)
    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Trojan Detector for Question & Answering Tasks.')

    # remember to switch this to 1 when making a container
    parser.add_argument('--is_submission', type=int, help='Flag to determine if this is a submission to the NIST server', default=0, choices=[0, 1])

    # trigger inversion variables
    parser.add_argument('--model_num',                    default=50, type=int, help="model number - only used if it's not a submission")                    
    parser.add_argument('--trigger_length',               default=5, type=int, help='How long do we want the trigger to be')
    parser.add_argument('--num_candidates',               default=50, type=int, help='How many candidates do we want to evaluate in each position')
    parser.add_argument('--lmbda',                        default=1.0, type=int, help='Weight on the evaluation loss')
    parser.add_argument('--q_trigger_insertion_location', default='end', type=str, help='Where in the question do we want to insert the trigger', choices=['start', 'end'])
    parser.add_argument('--c_trigger_insertion_location', default='end', type=str, help='Where in the context do we want to insert the trigger', choices=['start', 'end'])
    
    # do not change these ones
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/model.pt')
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default='./tokenizers/google-electra-small-discriminator.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.', default='./model/example_data')
  
    args = parser.parse_args()
    
    def modify_args_for_training(args):
        if args.is_submission == 0:
            metadata = pd.read_csv(join(TRAINING_FILEPATH, 'METADATA.csv'))

            id_str = str(100000000 + args.model_num)[1:]
            model_id = 'id-'+id_str

            args.model_filepath = join(TRAINING_FILEPATH, 'models', model_id, 'model.pt')
            args.examples_dirpath = join(TRAINING_FILEPATH, 'models', model_id, 'example_data')
            return args
        else:
            return args
    args = modify_args_for_training(args)

        # modify args and export LAMBDA
    args.eval_model_filepath = args.model_filepath
    args.trigger_insertion_locations = [args.q_trigger_insertion_location, args.c_trigger_insertion_location]
    global LAMBDA
    LAMBDA = args.lmbda

    trojan_detector(args)