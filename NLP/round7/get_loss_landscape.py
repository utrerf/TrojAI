import tools
import argparse
import playground
from os.path import join
import torch
import numpy as np



DEVICE = tools.DEVICE
BATCH_SIZE = 256


def loss_landscape(eval_model_filepath, tokenizer_filepath, 
                    result_filepath, model_num, examples_dirpath, is_training):
    '''LOAD MODELS, EMBEDDING, AND TOKENIZER'''
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

    class_list = tools.get_class_list(examples_dirpath)
    source_class = 1
    target_class = 7
    clean_class_list = [source_class, target_class]

    # trigger_token_ids = \
    #     tools.make_initial_trigger_tokens(is_random=False, initial_trigger_words="ok")
    trigger_token_ids = torch.tensor([0]).to(tools.DEVICE)
    trigger_length = len(trigger_token_ids)

    
    tools.update_logits_masks(class_list, clean_class_list, models['eval_model'])  
    temp_examples_dirpath = join('/'.join(clean_models_filepath[0].split('/')[:-1]), 'clean_example_data')
    vars, trigger_mask, source_class_token_locations =\
        playground.initialize_attack_for_source_class(temp_examples_dirpath, source_class, trigger_token_ids)
    best_k_ids = torch.tensor([list(tools.TOKENIZER.vocab.values())]).to(tools.DEVICE)
    loss_per_candidate = \
        playground.get_loss_per_candidate(models, vars, source_class_token_locations, 
            trigger_mask, trigger_token_ids, best_k_ids, len(trigger_token_ids)-1, 
            source_class, target_class, clean_class_list, class_list, is_testing=True)
    np.save(f'/scratch/utrerf/TrojAI/NLP/round7/loss_landscape_model_{model_num}/candidates',\
        np.array([i[0].detach().cpu().numpy() for i in loss_per_candidate]))
    np.save(f'/scratch/utrerf/TrojAI/NLP/round7/loss_landscape_model_{model_num}/losses',\
        np.array([i[1] for i in loss_per_candidate]))
    np.save(f'/scratch/utrerf/TrojAI/NLP/round7/loss_landscape_model_{model_num}/embeddings',\
        np.array([i[2] for i in loss_per_candidate]))

    
    

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Loss Landscape')

    parser.add_argument('--is_training', type=int, choices=[0, 1], 
                        help='Helps determine if we are training or testing.'\
                             ' If training just specify model number', 
                        default=1)
    parser.add_argument('--model_num', type=int, 
                        help='Model id number', 
                        default=190)
    parser.add_argument('--training_data_path', type=str, 
                        help='Folder that contains the training data', 
                        default=tools.TRAINING_DATA_PATH)
    parser.add_argument('--model_filepath', type=str, 
                        help='File path to the pytorch model file to be evaluated.', 
                        default='/scratch/data/TrojAI/round7-train-dataset/models/id-00000000/model.pt')
    parser.add_argument('--tokenizer_filepath', type=str, 
                        help='File path to the pytorch model (.pt) file containing the '\
                             'correct tokenizer to be used with the model_filepath.', 
                        default='/scratch/data/TrojAI/round7-train-dataset/tokenizers/MobileBERT-google-mobilebert-uncased.pt')
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
    parser.add_argument('--lmbda', type=float, 
                        help='Lambda used for the second term of the loss function to weigh the clean accuracy loss', 
                        default=1.)
    parser.add_argument('--num_candidates', type=int, 
                        help='number of candidates per token', 
                        default=1)   
    parser.add_argument('--beam_size', type=int, 
                    help='number of candidates per token', 
                    default=2)       
    parser.add_argument('--max_sentences', type=int, 
                    help='number of sentences to use', 
                    default=25)                      
    

    args = parser.parse_args()

    tools.LAMBDA=args.lmbda
    tools.NUM_CANDIDATES = args.num_candidates
    tools.BEAM_SIZE = args.beam_size
    tools.MAX_SENTENCES = args.max_sentences


    
    if args.is_training:
        args = tools.modify_args_for_training(args)

    loss_landscape(args.model_filepath, 
                    args.tokenizer_filepath, 
                    args.result_filepath, 
                    args.model_num,
                    args.examples_dirpath,
                    args.is_training)