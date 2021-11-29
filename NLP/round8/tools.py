import os
from os.path import join
import json
import torch
import transformers
import multiprocessing
import re
import datasets
from itertools import product, permutations
from copy import deepcopy
# valid word checker
import enchant

CPU = torch.device('cpu')

@torch.no_grad()
def load_config(model_filepath):
    '''
    Loads a .json config file called "config.json" in the model_filepath directory 
        and prints the source dataset filepath
    '''
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
    print('Source dataset name = "{}"'.format(config['source_dataset']))
    if 'data_filepath' in config.keys():
        print('Source dataset filepath = "{}"'.format(config['data_filepath']))
    return config


@torch.no_grad()
def load_models(config, args, CLEAN_TRAIN_MODELS_FILEPATH, CLEAN_TEST_MODELS_FILEPATH, DEVICE):
    # load all the models into a dictionary that contains eval, clean_train and clean_test models
    @torch.no_grad()
    def get_clean_model_filepaths(config, args, CLEAN_TRAIN_MODELS_FILEPATH, CLEAN_TEST_MODELS_FILEPATH, is_testing=False):
        key = f"{config['source_dataset'].lower()}_{config['model_architecture'].split('/')[-1]}_id"
        model_name = args.model_filepath.split('/')[-2]
        base_path = CLEAN_TRAIN_MODELS_FILEPATH
        max_models = None
        if is_testing:
            base_path = CLEAN_TEST_MODELS_FILEPATH
            max_models = args.max_test_models
        model_folders = [f for f in os.listdir(base_path) \
                            if (key in f and model_name not in f)][:max_models]
        clean_classification_model_paths = \
            [join(base_path, model_folder, 'model.pt') for model_folder in model_folders]       
        return clean_classification_model_paths
    clean_model_filepaths = {'train':get_clean_model_filepaths(config, args, CLEAN_TRAIN_MODELS_FILEPATH, CLEAN_TEST_MODELS_FILEPATH, is_testing=False),
                            'test' :get_clean_model_filepaths(config, args, CLEAN_TRAIN_MODELS_FILEPATH, CLEAN_TEST_MODELS_FILEPATH, is_testing=True)}
    if len(clean_model_filepaths['train']) == 0:
        clean_model_filepaths['train'].append(clean_model_filepaths['test'].pop(0))

    def load_model(model_filepath, map_location=DEVICE):
        classification_model = torch.load(model_filepath, map_location=map_location)
        classification_model.eval()
        return classification_model
    
    classification_model = load_model(args.eval_model_filepath)

    def load_clean_models(clean_model_filepath, map_location=DEVICE):
        clean_models = []
        for f in clean_model_filepath:
            clean_models.append(load_model(f, map_location=map_location))
        return clean_models

    models = {'eval': [classification_model],
                'clean_train': load_clean_models(clean_model_filepaths['train']),
                'clean_test': load_clean_models(clean_model_filepaths['test'], map_location=DEVICE)}
    # test
    assert len(models['eval'])==1,                          'wrong number of eval models'
    assert len(models['clean_train'])==1,                   'wrong number of clean train models'
    assert len(models['clean_test'])==args.max_test_models or \
            len(models['clean_test'])==args.max_test_models-1, 'wrong number of clean test models'
    return models, clean_model_filepaths

# get all the input embeddings
@torch.no_grad()
def get_all_input_id_embeddings(models):
    def get_embedding_weight(model):
        def find_word_embedding_module(model):
            word_embedding_tuple = [(name, module) 
                for name, module in model.named_modules() 
                if 'embeddings.word_embeddings' in name]
            assert len(word_embedding_tuple) == 1
            return word_embedding_tuple[0][1]
        word_embedding = find_word_embedding_module(model)
        word_embedding = deepcopy(word_embedding.weight).detach().to(CPU)
        word_embedding.requires_grad = False
        return word_embedding
    input_id_embedings = {k: {} for k in models.keys()}
    for model_type, model_list in models.items():
        for i, model in enumerate(model_list):
            input_id_embedings[model_type][i] = get_embedding_weight(model)
        input_id_embedings[model_type]['avg'] = torch.stack(list(input_id_embedings[model_type].values())).mean(dim=0)
    return input_id_embedings

# load the tokenizer that will convert text into input_ids (i.e. tokens) and viceversa
@torch.no_grad()
def load_tokenizer(is_submission, tokenizer_filepath, config):
    if is_submission:
        tokenizer = torch.load(tokenizer_filepath)
    else:
        model_architecture = config['model_architecture']
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_architecture, use_fast=True)
    return tokenizer


TOOL = enchant.Dict("en_US")
@torch.no_grad()
def check_word(tup):
    ix, cand = tup
    if TOOL.check(cand) == True:
        return ix

@torch.no_grad()
def get_most_changed_embeddings(input_id_embeddings, tokenizer, DEVICE, k=10000):
    total_cos = torch.nn.CosineSimilarity(dim=0, eps=1e-15)
    total_cos_sim_dict = {}
    for i, val in input_id_embeddings['clean_test'].items():
        total_cos_sim_dict[i] = -total_cos(input_id_embeddings['eval']['avg'].flatten(), val.flatten())

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-15)
    cos_sim_dict = {}
    for i, val in input_id_embeddings['clean_test'].items():
        cos_sim_dict[i] = -cos(input_id_embeddings['eval']['avg'], val)
    min_cos_sim = torch.stack(list(cos_sim_dict.values())).min(dim=0)[0]
    smallest_values, top_ids = torch.topk(min_cos_sim,k)
    top_ids = top_ids.to(DEVICE)
    smallest_values = smallest_values[:5]
    top_ids_to_tokens = {top_id:tokenizer.convert_ids_to_tokens([top_id])[0] for top_id in top_ids}            
    top_ids = [top_id for top_id, token in top_ids_to_tokens.items() if re.match(r'[#]*\w([A-Za-z]+)[#]*', token) is not None][:100]

    all_suffixes = [i for i in top_ids if '##' in tokenizer.convert_ids_to_tokens([i])[0]]
    suffixes = all_suffixes[:5]
    prefixes = [i for i in top_ids if i not in all_suffixes]

    suffixes_combinations = []
    for i in range(1, 3):
        new_combination = list(permutations(suffixes, i))
        new_combination = [list(i) for i in new_combination]
        suffixes_combinations += new_combination
    
    candidates = []
    for p in prefixes:
        p_copy = deepcopy(p)
        candidates += [[p_copy]+i for i in suffixes_combinations]
    decoded_candidates = tokenizer.batch_decode(candidates)
    
    pool_obj = multiprocessing.Pool()

    composed_words = pool_obj.map(check_word, [(ix, cand) for ix, cand in enumerate(decoded_candidates)])
    composed_words = [candidates[i] for i in composed_words if i is not None]

    return {'single_token_words':prefixes[:8],
            'multi_token_words' :composed_words,
            'smallest_values': smallest_values,
            'total_similarity': total_cos_sim_dict.values()}

# load the dataset with text containing questions and answers
@torch.no_grad()
def load_dataset(examples_dirpath, scratch_dirpath, clean_model_filepaths=None, more_clean_data=False):
    clean_fns = []
    if more_clean_data:
        for model_type_paths in clean_model_filepaths.values():
            clean_examples_dirpath_list = ['/'.join(v.split('/')[:-1]+['example_data']) for v in model_type_paths]
            for dirpath in clean_examples_dirpath_list:
                clean_fns += [os.path.join(dirpath, fn) for fn in os.listdir(dirpath) if (fn.endswith('.json') and 'clean'in fn)]

    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if (fn.endswith('.json') and 'clean'in fn)]
    fns.sort()
    
    examples_filepath_list = fns + clean_fns

    dataset_list = []
    for examples_filepath in examples_filepath_list:
        # Load the examples
        dataset_list.append(datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache')))

    return datasets.concatenate_datasets(dataset_list)        


# tokenize the dataset to be able to feed it to the NLP model during inference
@torch.no_grad()
def tokenize_for_qa(tokenizer, dataset):

    question_column_name, context_column_name, answer_column_name  = "question", "context", "answers"
    
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    # max_seq_length = min(tokenizer.model_max_length, 384)
    max_seq_length = min(tokenizer.model_max_length, 200)
    
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
        
        # initialize lists
        var_list = ['question_start_and_end', 'context_start_and_end', 
                    'train_clean_baseline_likelihoods', 'train_eval_baseline_likelihoods', 
                    'test_clean_baseline_likelihoods', 'test_eval_baseline_likelihoods', 
                    'train_clean_answer_likelihoods', 'train_eval_answer_likelihoods', 
                    'test_clean_answer_likelihoods', 'test_eval_answer_likelihoods', 
                    'answer_start_and_end', 'repeated']
        for var_name in var_list:
            tokenized_examples[var_name] = []
        
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        already_included_samples = set()
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
            if sample_ix in already_included_samples:
                tokenized_examples['repeated'].append(True)
            else:
                tokenized_examples['repeated'].append(False)
            already_included_samples.add(sample_ix)
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
                    while token_answer_start_ix < len(offsets) and offsets[token_answer_start_ix][0] <= start_char:
                        token_answer_start_ix += 1
                    while offsets[token_answer_end_ix][1] >= end_char:
                        token_answer_end_ix -= 1
                    set_answer_start_and_end_to_ixs(token_answer_start_ix-1, token_answer_end_ix+1)
            
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_ix else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
            
            for train_test, eval_clean in product(['train', 'test'], ['eval', 'clean']):
                tokenized_examples[f'{train_test}_{eval_clean}_baseline_likelihoods'].append(torch.zeros(1))
            
            for train_test, eval_clean in product(['train', 'test'], ['eval', 'clean']):
                tokenized_examples[f'{train_test}_{eval_clean}_answer_likelihoods'].append(torch.zeros(1))


        return tokenized_examples
    
    tokenized_dataset = dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=10,
        remove_columns=dataset.column_names,
        keep_in_memory=True)

    tokenized_dataset = tokenized_dataset.remove_columns(['offset_mapping'])
    assert len(tokenized_dataset) > 0

    return tokenized_dataset


# select the sentences that have an answer in the context (i.e. cls is not the answer)
@torch.no_grad()
def select_examples_with_an_answer_in_context(tokenized_dataset, tokenizer):
    answer_starts = torch.tensor(tokenized_dataset['answer_start_and_end'])[:, 0]
    non_cls_answer_indices = (~torch.eq(answer_starts, tokenizer.cls_token_id)).nonzero().flatten()
    return tokenized_dataset.select(non_cls_answer_indices)


@torch.no_grad()
def select_unique_inputs(tokenized_dataset):
    unique_ixs_ids = torch.tensor(tokenized_dataset['input_ids']).unique(dim=0, return_inverse=True)[1].flatten()
    seen = set()
    unique_ixs = []
    for source_ix, target_ix in enumerate(unique_ixs_ids):
        if target_ix.item() not in seen:
            seen.add(target_ix.item())
            unique_ixs.append(source_ix)
        
    tokenized_dataset = tokenized_dataset.select(unique_ixs)
    unique_ixs_source = (~(torch.tensor(tokenized_dataset['repeated']).bool())).nonzero().flatten()
    return tokenized_dataset.select(unique_ixs_source)