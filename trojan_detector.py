import torch
from torch.utils.data import DataLoader
import argparse
import tools
import numpy as np
import pandas as pd
import custom_transforms
from torchvision import transforms
from robustness import attacker
import chop.optim
import re
# import sys
# sys.path.append('PyHessian')
# from pyhessian import hessian
import classifier
import joblib

def trojan_detector(model_filepath, result_filepath, scratch_dirpath, 
                    examples_dirpath, example_img_format='png', is_train=False):
    
    torch.set_num_threads(1)
    tools.set_seeds(123)
    
    # load dataset and model
    dataset = tools.TrojAIDataset(examples_dirpath)
    model = torch.load(model_filepath) 
    model = tools.CustomAdvModel(model)
    model.cuda().eval()
    
    scores = {} 
    
    # get num_parameters and classes
    model_info = tools.get_model_info(model, dataset)
    for key, val in model_info.items():
        scores[key] = val

    # titration
    for noise_level in [1.6]:
        transform = custom_transforms.TITRATION_TRANSFORM(noise_level)
        score = f'noise_level_{noise_level}'
        scores = tools.transform_scores(model, dataset, transform, scores, score, num_iterations=10)

    # erase
   # for erase_probability in [1]:
   #     transform = custom_transforms.ERASE_TRANSFORM(erase_probability) 
   #     score = f'erase_probability_{erase_probability}' 
   #     scores = tools.transform_scores(model, dataset, transform, scores, score, num_iterations=40)
   # 
    # TODO: decide where to call adv_scores for chop constraints

    # adversarial
    adv_dataset = tools.MadryDataset(None, num_classes=model_info['num_classes'])
    adv_model = attacker.AttackerModel(model, adv_dataset)
    hessian_datasets = {'natural':dataset}
    hessian_list = [.15, .5, 2., 8., 16.]
    constraints_to_eps = {
        'inf' : [.15],
        '2'   : [.5, 2., 4., 8., 10., 16.],
        'groupLasso': [.1, .5, 1.5, 2., 4., 10.],
        'tracenorm': [.1, .5, 1., 5.]
    }
    for constraint, eps_list in constraints_to_eps.items():
        for eps in eps_list:
            score = f'{constraint}_eps_{eps}'
            flag = eps in hessian_list
            if constraint in ['groupLasso', 'tracenorm']:
                adversary_alg = chop.optim.minimize_frank_wolfe
            else:
                adversary_alg = None
            scores, hessian_datasets[score] = tools.adv_scores(adv_model, dataset, scores, score, constraint=constraint, eps=float(eps), 
                                                                 batch_size=32, iterations=7, compute_top_eigenvalue=flag,
                                                                 adversary_alg=adversary_alg)
    
    # hessian
    del adv_dataset, adv_model
    torch.cuda.empty_cache()
    for score, hessian_dataset in hessian_datasets.items():
        if hessian_dataset is not None:
            del model
            torch.cuda.empty_cache()
            model = torch.load(model_filepath) 
            model.cuda().eval()
            # scores = tools.compute_top_eigenvalue(model, hessian_dataset, scores, score, batch_size=40, max_iter=20)
            scores = tools.compute_grad_l2_norm(model, hessian_dataset, scores, score, batch_size=40)

    results_df = pd.DataFrame(scores, [0])
    
    if is_train:
        # save results
        model_id = re.findall(r'id-(\d+)/', model_filepath)[0]
        results_df.to_csv(f'results/id-{model_id}.csv')
    else:
        # make prediction
        clf_filename = '/model.joblib'
        features_filename = '/features.csv'

        clf = joblib.load(clf_filename)
        features = list(pd.read_csv(features_filename).columns)
        
        trojan_probability = classifier.make_prediction(results_df)
        with open(result_filepath, 'w') as fh:
            fh.write("{}".format(trojan_probability))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='/scratch/utrerf/round2/models/id-00000001/model.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='temp')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='/scratch/utrerf/round2/models/id-00000001/example_data') 
    parser.add_argument('--is_train', type=bool, help='If True, then it saves results to csv and doesnt do inference.', default=False)

    args = parser.parse_args()
    trojan_detector(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath, args.is_train)
