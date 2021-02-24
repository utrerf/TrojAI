import os
import numpy as np
import pandas as pd
import copy
import transformers
import jsonpickle
import argparse
import json
import model_factories
import torch
import tools
import cv_tools
from robustness import attacker
import re
import chop
import constants

import warnings
warnings.filterwarnings("ignore")


def trojan_detector(model_filepath, cls_token_is_first, tokenizer_filepath, embedding_filepath, 
                                  result_filepath, scratch_dirpath, examples_dirpath, train_test):
    
    use_amp = True  # attempt to use mixed precision to accelerate embedding conversion process
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load the model, examples, embeddings
    model = torch.load(model_filepath)
    model.cuda().train()
    param_list = []
    for n, p in model.named_parameters():
        print(f'parameter: {n}')
        print(f'requires grad: {p.requires_grad}')
        param_list.append(p)

    char_df = pd.read_csv('32_char_df.csv.gz')
    text_list = tools.read_examples_into_list(examples_dirpath)
    
    embedding = torch.load(embedding_filepath)
    embedding_flavor = tools.determine_embedding_flavor(embedding) 
   
    # 2. Deduce text dataset name, load pre-computed embeddings and labels in dataset and loader
    dataset_name = tools.determine_dataset_name(text_list, char_df)
    
    embeddings_base_path = constants.datasets_base_path
    dataset = tools.TrojAIDataset(dataset_name, embedding_flavor, embeddings_base_path, train_test) 
    # batch_size, num_workers = 16384, 0
    # loader = DataLoader(ds, batch_size=batch_size, sampler=tools.make_weighted_sampler(ds),
    #                                              num_workers=num_workers, pin_memory=True)

    # 3. Apply adversarial perturbations in increasing levels
    scores = {}
    
    # TODO: Avoid hardcoding num_classes
    adv_dataset = cv_tools.MadryDatasetNLP(None, num_classes=2)
    adv_model = attacker.AttackerModel(model, adv_dataset)
    constraints_to_eps = {
       '2'   : [0.25, 1., 4., 16.],
       #'tracenorm': 10 ** np.linspace(-3, 3, num=10),
       #'groupLasso': 10 ** np.linspace(-5, -1, num=10)
    }
    adv_datasets = {}
    for constraint, eps_list in constraints_to_eps.items():
        for eps in eps_list:
            score = f'{constraint}_eps_{eps}'
            if constraint in ['groupLasso', 'tracenorm']:
                adversary_alg = chop.optim.minimize_frank_wolfe
            else:
                adversary_alg = None
            scores, adv_datasets[score] = cv_tools.adv_scores(adv_model, dataset, scores, score, 
                                                             constraint=constraint, eps=float(eps), 
                                                             batch_size=32768, iterations=20, 
                                                             adversary_alg=adversary_alg)
    
   
    
    # 4. Wrap up
    results_df = pd.DataFrame(scores, [0])
    if train_test=='train':
        # save predictions
        model_id = re.findall(r'id-(\d+)/', model_filepath)[0]
        results_df.to_csv(f'results/id-{model_id}.csv', index=None)
    
    else:
        # Test scratch space
        with open(os.path.join(scratch_dirpath, 'test.txt'), 'w') as fh:
            fh.write('this is a test')

        # TODO: Get a classifier to compute the trojan probability
        trojan_probability = np.random.rand()
        print('Trojan Probability: {}'.format(trojan_probability))

        with open(result_filepath, 'w') as fh:
            fh.write("{}".format(trojan_probability))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_filepath', type=str, 
                        help='File path to the pytorch model file to be evaluated.', 
                        default='./model/model.pt')
    parser.add_argument('--cls_token_is_first', type=bool, 
                        help='Whether the first embedding token should be used as the'+
                             ' summary of the text sequence, or the last token.', 
                        default=True)
    parser.add_argument('--tokenizer_filepath', type=str, 
                        help='File path to the pytorch model file to be evaluated.', 
                        default='./model/tokenizer.pt')
    parser.add_argument('--embedding_filepath', type=str, 
                        help='File path to the pytorch model file to be evaluated.', 
                        default='./model/embedding.pt')
    parser.add_argument('--result_filepath', type=str, 
                        help='File path to the file where output result should be written.'+
                             ' After execution this file should contain a single line with'+
                                             ' a single floating point trojan probability.', 
                        default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, 
                         help='File path to the folder where scratch disk space exists. '+
                              'This folder will be empty at execution start and will be '+ 
                              'deleted at completion of execution.', 
                         default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, 
                         help='File path to the folder of examples which might be useful'+
                              ' for determining whether a model is poisoned.', 
                         default='./model/clean_example_data')
    parser.add_argument('--train_test', type=str, 
                         help='If train, we save features. Otherwise, we make a prediction',
                         default='test')
    
    args = parser.parse_args()
    print(args)

    trojan_detector(args.model_filepath, args.cls_token_is_first, args.tokenizer_filepath, 
                    args.embedding_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath,
                    args.train_test)

