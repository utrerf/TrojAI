# external libraries
import argparse
import time
from os.path import join
from copy import deepcopy
import numpy as np
import pandas as pd
from random import randint
import torch
import torch.optim as optim
from torch.cuda.amp import autocast
from datasets.utils.logging import set_verbosity_error
set_verbosity_error()
import warnings
warnings.filterwarnings("ignore")

# our files
import tools
from filepaths import SANDBOX_TRAINING_FILEPATH,    SANDBOX_CLEAN_TRAIN_MODELS_FILEPATH,    SANDBOX_CLEAN_TEST_MODELS_FILEPATH
from filepaths import SUBMISSION_TRAINING_FILEPATH, SUBMISSION_CLEAN_TRAIN_MODELS_FILEPATH, SUBMISSION_CLEAN_TEST_MODELS_FILEPATH

''' CONSTANTS '''
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CPU = torch.device('cpu')

EXTRACTED_GRADS = {'eval':[], 'clean_train':[]}

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

    # load config
    if args.is_manual_config:
        config = {
            'source_dataset': args.source_dataset,
            'model_architecture': args.model_architecture
        }
    else:
        config = tools.load_config(args.eval_model_filepath)

    # load all models and get the clean_model_filepaths, which is used to get additional clean examples
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

    # get the word_embeddings for all the models
    input_id_embeddings = tools.get_all_input_id_embeddings(models)

    tokenizer = tools.load_tokenizer(args.is_submission, args.tokenizer_filepath, config)

    # get the candidate pool of tokens to start from at random initialization
    total_cand_pool = tools.get_most_changed_embeddings(input_id_embeddings, tokenizer, DEVICE)

    def only_keep_avg_input_id_embeddings(input_id_embeddings):
        return {model_type:{'avg':input_id_embeddings[model_type]['avg']} for model_type in list(input_id_embeddings.keys())}
    input_id_embeddings = only_keep_avg_input_id_embeddings(input_id_embeddings)

    # load and transform qa dataset
    dataset = tools.load_qa_dataset(args.examples_dirpath, args.scratch_dirpath, clean_model_filepaths, more_clean_data=args.more_clean_data)
    print(f'dataset length: {len(dataset)}')
    tokenized_dataset = tools.tokenize_for_qa(tokenizer, dataset)
    tokenized_dataset = tools.select_examples_with_an_answer_in_context(tokenized_dataset, tokenizer)
    tokenized_dataset = tools.select_unique_inputs(tokenized_dataset)

    df = pd.DataFrame()
    # these are the behavior|insterions specific to QA
    for behavior, insertion in [('self', 'both'), ('cls', 'both'), ('self', 'context'), ('cls', 'context'), ('cls', 'question')]:
        best_test_loss = None
        start_time = time.time()
        args.trigger_behavior, args.trigger_insertion_type = behavior, insertion

        # add a dummy trigger into input_ids, attention_mask, and token_type as well as provide masks for loss calculations
        triggered_dataset = tools.get_triggered_dataset(args, DEVICE, tokenizer, tokenized_dataset)

        # insert trigger and populate baselines
        def insert_trigger_and_populate_baselines():
            tools.insert_new_trigger(args, triggered_dataset, torch.tensor([tokenizer.pad_token_id]*args.trigger_length, device=DEVICE).long())
            # zero out attention on trigger
            tools.insert_new_trigger(args, triggered_dataset, torch.zeros(args.trigger_length, device=DEVICE).long(), where_to_insert='attention_mask')

            # train loss to get train baseline
            tools.compute_loss(args, models, triggered_dataset, args.batch_size, DEVICE, LAMBDA, with_gradient=False, populate_baselines=True)
            
            # test loss to get train baseline
            models['clean_test'] = [model.to(DEVICE, non_blocking=True) for model in models['clean_test']]
            tools.compute_loss(args, models, triggered_dataset, args.batch_size, DEVICE, LAMBDA, with_gradient=False, train_or_test='test', populate_baselines=True)
            models['clean_test'] = [model.to(CPU, non_blocking=True) for model in models['clean_test']]

            # add back attention
            tools.insert_new_trigger(args, triggered_dataset, torch.ones(args.trigger_length, device=DEVICE).long(), where_to_insert='attention_mask')
        insert_trigger_and_populate_baselines()
        
        triggered_dataset = tools.take_best_k_inputs(triggered_dataset)

        def put_embeds_on_device(device=DEVICE):
            input_id_embeddings['eval']['avg'] = input_id_embeddings['eval']['avg'].to(device, non_blocking=True)
            input_id_embeddings['clean_train']['avg'] = input_id_embeddings['clean_train']['avg'].to(device, non_blocking=True)
        put_embeds_on_device()

        for i in range(args.num_random_tries):
            # get and insert new trigger
            num_non_random_tries = args.num_random_tries//2 if 'electra' in config['model_architecture'] else 2*(args.num_random_tries//3)
            init_fn = args.trigger_init_fn if i < num_non_random_tries else 'random'
            new_trigger = tools.initialize_trigger(args, init_fn, total_cand_pool, tokenizer, DEVICE) 
            tools.insert_new_trigger(args, triggered_dataset, new_trigger)

            old_trigger, n_iter = torch.tensor([randint(0,20000) for _ in range(args.trigger_length)]).to(DEVICE), 0
            with autocast():
                # main trigger inversion loop for a given random start     
                while not torch.equal(old_trigger, new_trigger) and n_iter < args.max_iter:
                    n_iter += 1
                    old_trigger, old_loss = deepcopy(new_trigger), tools.compute_loss(args, models, triggered_dataset, args.batch_size, DEVICE, LAMBDA, with_gradient=True)

                    @torch.no_grad()
                    def find_best_k_candidates_for_each_trigger_token(num_candidates, tokenizer):    
                        '''
                        equation 2: (embedding_matrix - trigger embedding)T @ trigger_grad
                        '''
                        embeds_shape = [len(triggered_dataset['input_ids']), -1, input_id_embeddings['eval']['avg'].shape[-1]]

                        def get_mean_trigger_grads(eval_or_clean):
                            concat_grads = torch.cat(EXTRACTED_GRADS[eval_or_clean])
                            grads_list = []
                            if args.trigger_insertion_type in ['context', 'both']:
                                mean_context_grads_over_inputs = concat_grads[triggered_dataset['c_trigger_mask']].view(embeds_shape).mean(dim=0)
                                grads_list.append(mean_context_grads_over_inputs)
                            if args.trigger_insertion_type in ['question', 'both']:
                                mean_question_grads_over_inputs = concat_grads[triggered_dataset['q_trigger_mask']].view(embeds_shape).mean(dim=0)
                                grads_list.append(mean_question_grads_over_inputs)
                            return torch.stack(grads_list).mean(dim=0)                
                        eval_mean_trigger_grads = get_mean_trigger_grads('eval')
                        clean_train_mean_trigger_grads = get_mean_trigger_grads('clean_train')
                        
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
                        return best_k_ids
                    candidates = find_best_k_candidates_for_each_trigger_token(args.num_candidates, tokenizer)
                    candidates = tools.add_candidate_variations_to_candidates(old_trigger, args, candidates)

                    def clear_all_model_grads(models):
                        for model_type, model_list in models.items():
                            for model in model_list:
                                optimizer = optim.Adam(model.parameters())
                                optimizer.zero_grad(set_to_none=True)
                        for model_type in EXTRACTED_GRADS.keys():
                            EXTRACTED_GRADS[model_type] = []
                    clear_all_model_grads(models)
 
                    new_loss, new_trigger = tools.evaluate_and_pick_best_candidate(args, models, DEVICE, LAMBDA, old_loss, old_trigger, triggered_dataset, candidates, args.beam_size)
                    tools.insert_new_trigger(args,triggered_dataset, new_trigger)
                    
                    if args.is_submission == False:
                        tools.print_results_of_trigger_inversion_iterate(n_iter, tokenizer, old_loss, new_loss, old_trigger, new_trigger, LAMBDA, triggered_dataset)

                    del old_loss, candidates

                new_test_loss = tools.get_new_test_loss(args, models, triggered_dataset, DEVICE, LAMBDA)
            if best_test_loss is None or best_test_loss['trigger_inversion_loss'] > new_test_loss['trigger_inversion_loss']:
                best_trigger, best_train_loss, best_test_loss = deepcopy(new_trigger), deepcopy(new_loss), deepcopy(new_test_loss)
            
            if best_test_loss['trigger_inversion_loss'] < args.stopping_threshold:
                break
            torch.cuda.empty_cache()
        
        df = tools.add_results_to_df(args, df, tokenizer, best_trigger, best_train_loss, best_test_loss, total_cand_pool, triggered_dataset, start_time)

    if args.is_submission:
        tools.submit_results()
    else:
        tools.save_results(df, args)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Trojan Detector With Trigger Inversion for Question & Answering Tasks.')

    def add_all_args(parser):
        # general_args: remember to switch is_submission to 1 when making a container
        parser.add_argument('--is_submission',      dest='is_submission',   action='store_true',  help='Flag to determine if this is a submission to the NIST server',  )
        parser.add_argument('--is_manual_config',   dest='is_manual_config',action='store_true',  help='Flag to determine if we are entering a manual config file instead of search for config.json in the model folder',  )
        # TODO: Update the source_dataset and model_architecture
        parser.add_argument('--source_dataset',     default=None,  type=str,   help='What is the dataset',            choices=['squad_v2', 'subjqa'])
        parser.add_argument('--model_architecture', default=None,  type=str,   help='What is the model architecture', choices=['discrete', 'relaxed'])
        # END TODO
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
        parser.add_argument('--num_random_tries',             default=1,          type=int,   help='How many random starts do we try')
        parser.add_argument('--trigger_length',               default=7,           type=int,   help='How long do we want the trigger to be')
        parser.add_argument('--lmbda',                        default=2.,          type=float, help='Weight on the clean loss')
        parser.add_argument('--temperature',                  default=1.,          type=float, help='Temperature parameter to divide logits by')
        parser.add_argument('--stopping_threshold',           default=0.01,        type=float, help='if trigger inversion test loss is lower than this value we do not randomly initialize again')
        

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
    parser.set_defaults(is_submission=False, more_clean_data=True, calculate_alpha=False)
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