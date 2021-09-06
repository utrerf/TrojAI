'''
NOTES
- Assume that the code will not pass answer position information
- Loss function will either add up or max logits
- Understand how to combine loss from start and end
- The trigger should be introduced in both the question and the answer
    - Do we need to be careful when introducing the trigger to avoid adding it to examples that do not contain the context?
QUESTIONS
- Why do we pass the *answer* start/end position instead of the *context* start/end position to the model during the forward pass? 
    See the prepare_train_features function:
        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
        # Note: we could go after the last offset if the answer is the last word (edge case).
        while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
            token_start_index += 1
        tokenized_examples["start_positions"].append(token_start_index - 1)
        while offsets[token_end_index][1] >= end_char:
            token_end_index -= 1
        tokenized_examples["end_positions"].append(token_end_index + 1)
- Discover what is the point of token_type_ids in distilbert
    encoded_dict['token_type_ids']
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
- Can we just add to the current context without overflowing? Let's assume yes.
'''
import os
from os.path import join

import datasets
import pandas as pd
import torch
import transformers
import json
from copy import deepcopy

import warnings

import utils_qa

warnings.filterwarnings("ignore")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')
TRAINING_FILEPATH = '/scratch/data/TrojAI/round8-train-dataset'
CLEAN_MODELS_FILEPATH_TRAIN = '/scratch/utrerf/TrojAI/NLP/round8/clean_models_train'
CLEAN_MODELS_FILEPATH_TEST = '/scratch/utrerf/TrojAI/NLP/round8/clean_models_test'
EXTRACTED_GRADS = {'eval':[], 'clean':[]}


# The inferencing approach was adapted from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
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
        for clean_model in models['clean_train']:
            var_list = get_fwd_var_list(models['eval'])
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
        dataset_instance['cls_mask'] = cls_mask.long()

        return dataset_instance
    
    triggered_dataset = tokenized_dataset.map(
        initialize_dummy_trigger_helper,
        batched=False,
        num_proc=2,
        keep_in_memory=True)

    triggered_dataset = triggered_dataset.remove_columns([f'{v}_start_and_end' for v in ['question', 'context', 'answer']])
    triggered_dataset.set_format('pt', columns=triggered_dataset.column_names)

    return triggered_dataset

    
def insert_new_trigger(triggered_dataset, trigger):
    trigger = torch.tensor(trigger)

    def insert_new_trigger_helper(dataset_sample):
        dataset_sample['input_ids'][dataset_sample['q_trigger_mask']] = trigger
        dataset_sample['input_ids'][dataset_sample['c_trigger_mask']] = trigger
        return dataset_sample

    triggered_dataset = triggered_dataset.map(
        insert_new_trigger_helper,
        batched=False,
        num_proc=1,
        keep_in_memory=True)

    # triggered_dataset.set_format('pt', columns=triggered_dataset.column_names)
    
    return triggered_dataset


def print_inputs(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath):
    for v, vn in zip([model_filepath,   tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath], 
                     ['model',          'tokenizer',        'result',        'scratch',       'examples']):
        print(f'{v}_filepath = {vn}')


def load_dataset(examples_dirpath, scratch_dirpath):
    # clean example inference
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
    fns.sort()
    examples_filepath = fns[0]

    # Load the examples
    # TODO The cache_dir is required for the test server since /home/trojai is not writable and the default cache locations is ~/.cache
    dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))
    return dataset


def load_tokenizer(is_submission, tokenizer_filepath, config):
    if is_submission:
        tokenizer = torch.load(tokenizer_filepath)
    else:
        model_architecture = config['model_architecture']
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_architecture, use_fast=True)
    return tokenizer


def load_config(model_filepath):
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
    print('Source dataset name = "{}"'.format(config['source_dataset']))
    if 'data_filepath' in config.keys():
        print('Source dataset filepath = "{}"'.format(config['data_filepath']))
    return config


def load_all_models(eval_model_filepath, config):
    def load_model(model_filepath):
        classification_model = torch.load(model_filepath, map_location=DEVICE)
        classification_model.eval()
        return classification_model
    
    classification_model = load_model(eval_model_filepath)

    def get_clean_model_filepaths(config, is_testing=False):
        key = f"{config['source_dataset'].lower()}_{config['model_architecture'].split('/')[-1]}"
        model_name = config['output_filepath'].split('/')[-1]
        base_path = CLEAN_MODELS_FILEPATH_TRAIN
        if is_testing:
            base_path = CLEAN_MODELS_FILEPATH_TEST
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

    return {'eval': classification_model,
            'clean_train': clean_models_train,
            'clean_test': clean_models_test}


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

    add_hooks(models['eval'], is_clean=False)

    for clean_model in models['clean_train']:
        add_hooks(clean_model, is_clean=True)


def get_fwd_var_list(model):
    var_list = ['input_ids', 'attention_mask']
    if ('distilbert' not in model.name_or_path) and ('bart' not in model.name_or_path):
            var_list += ['token_type_ids']
    return var_list


def trigger_inversion_loss(input_ids, output, batch):
    
    softmax = torch.nn.Softmax()

    def clean_loss():
        def clean_loss_pos(pos, logits):
            assert pos == 'start' or pos == 'end' 
            pos_ix = 0
            if pos == 'end':
                pos_ix = 1

            logits = output[f'{pos}_logits']
            valid_outputs_mask = batch['context_mask'] | batch['cls_mask']
            valid_likelihoods = softmax(logits[valid_outputs_mask])
            
            valid_cls_mask = batch['cls_mask'][valid_outputs_mask]
            cls_likelihood = valid_likelihoods[valid_cls_mask]
            net_cls_likelyhood = max(cls_likelihood - batch['clean_cls_likelihoods'][pos_ix], 0)

            valid_trigger_mask = batch['c_trigger_mask'][valid_outputs_mask]
            trigger_likelihood = valid_likelihoods[valid_cls_mask].sum(dim=0)
            base_trigger_likelihood = trigger_likelihood.shape()[0]/trigger_likelihood.shape()[-1]
            net_trigger_likelyhood = max(trigger_likelihood - base_trigger_likelihood, 0)          
            
            return max(-torch.log(1-max(net_cls_likelyhood, net_trigger_likelyhood)))
        
        # clean_losses = {'start', 'end'}
        # for pos in ['start', 'end']:
            
        #     for logits in output['clean']['f{pos}_logits']:

            
        

        # return (clean_loss_pos('start')+clean_loss_pos('end'))/2

    # def evaluation_loss():

    #     return (eval_loss_pos('start')+eval_loss_pos('end'))/2


    return NotImplementedError


def trojan_detector(eval_model_filepath, trigger_length, trigger_insertion_locations, 
                    tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath, is_submission):
    print_inputs(eval_model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath)

    # load the config file with details about the eval model
    config = load_config(eval_model_filepath)

    # load all models as a dictionary that includes eval, clean_train, and clean_test models
    models = load_all_models(eval_model_filepath, config)   
    add_hooks_to_all_models(models)

    # load the tokenizer that converts text into token_ids 
    tokenizer = load_tokenizer(is_submission, tokenizer_filepath, config)

    # load dataset and tokenize it
    dataset = load_dataset(examples_dirpath, scratch_dirpath)
    tokenized_dataset = tokenize_for_qa(tokenizer, dataset, models)
    
    # select non_cls_examples
    answer_starts = torch.tensor(tokenized_dataset['answer_start_and_end'])[:, 0]
    non_cls_answer_indices = (~torch.eq(answer_starts, tokenizer.cls_token_id))\
                                                             .nonzero().flatten()
    tokenized_dataset = tokenized_dataset.select(non_cls_answer_indices)

    # add a dummy trigger and then substitute it for a new trigger
    triggered_dataset = initialize_dummy_trigger(tokenized_dataset, tokenizer, trigger_length, trigger_insertion_locations)
    new_trigger = [100] * trigger_length
    triggered_dataset = insert_new_trigger(triggered_dataset, new_trigger)

    # make a dataloader 
    dataloader = torch.utils.data.DataLoader(triggered_dataset, batch_size=25)

    # delete unused variables
    del dataset, tokenized_dataset

    models['eval'].eval()
    for _, batch in enumerate(dataloader):
        # get the var_list for the model fwd function
        var_list = get_fwd_var_list(models['eval'])

        output = {}
        output['eval'] = models['eval'](**{v:batch[v].to(DEVICE) for v in var_list})
        for clean_model in models['clean_train']:
            output['clean'] = models['clean_train'](**{v:batch[v].to(DEVICE) for v in var_list})

        loss = trigger_inversion_loss(output, batch)
        loss.backward()


def modify_args_for_training(args):
    metadata = pd.read_csv(join(TRAINING_FILEPATH, 'METADATA.csv'))

    id_str = str(100000000 + args.model_num)[1:]
    model_id = 'id-'+id_str

    args.model_filepath = join(TRAINING_FILEPATH, 'models', model_id, 'model.pt')
    args.examples_dirpath = join(TRAINING_FILEPATH, 'models', model_id, 'example_data')
    return args


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_num', type=int, help="model number - only used if it's not a submission", 
                        default=50)
    parser.add_argument('--trigger_length', type=int, help='How long do we want the trigger to be', 
                        default=5)
    parser.add_argument('--q_trigger_insertion_location', type=str, help='Where in the question do we want to insert the trigger', choices=['start', 'end'],
                        default='end')
    parser.add_argument('--c_trigger_insertion_location', type=str, help='Where in the context do we want to insert the trigger', choices=['start', 'end'],
                        default='end')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/model.pt')
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default='./tokenizers/google-electra-small-discriminator.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.', default='./model/example_data')
    parser.add_argument('--is_submission', type=int, help='Flag to determine if this is a submission to the NIST server', default=0, choices=[0, 1])

    args = parser.parse_args()

    if args.is_submission == 0:
        args = modify_args_for_training(args)

    trojan_detector(args.model_filepath, 
                    args.trigger_length,
                    [args.q_trigger_insertion_location, args.c_trigger_insertion_location],
                    args.tokenizer_filepath, 
                    args.result_filepath, 
                    args.scratch_dirpath, 
                    args.examples_dirpath,
                    args.is_submission)