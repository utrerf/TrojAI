import torch
import argparse
import tools
import numpy as np
import pandas as pd
import custom_transforms
from torchvision import transforms
from robustness import attacker
import re
import sys
sys.path.append('/home/ubuntu/utrerf/PyHessian')
from pyhessian import hessian


def trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, example_img_format='png'):
    # load dataset and model
    tools.set_seeds(123)
    dataset = tools.TrojAIDataset(examples_dirpath)
    
    model = torch.load(model_filepath) 
    model_info = tools.get_model_info(model)
    model.cuda().eval()
    
    scores = {}
    
    # titration
    for noise_level in [.1, .4, .8, 1.6]:
        transform = custom_transforms.TITRATION_TRANSFORM(noise_level)
        scores[f't_score_noise_level_{noise_level}'] = tools.get_transform_score(model, dataset, transform, num_iterations=10)
    
    # erase
    for erase_probability in [1]:
        transform = custom_transforms.ERASE_TRANSFORM(erase_probability) 
        scores[f't_score_erase_probability_{erase_probability}'] = tools.get_transform_score(model, dataset, 
                                                                                             transform, num_iterations=20)
    # adversarial
    adv_dataset = tools.MadryDataset(None, num_classes=model_info['num_classes'])
    adv_model = attacker.AttackerModel(tools.CustomAdvModel(model), adv_dataset)
    for eps in [.005, 0.015, 0.075, .15]:
        scores[f'adv_score_linf_eps_{eps}'] = tools.get_adv_score(adv_model, dataset, constraint='inf', eps=eps, iteratons=7)
    for eps in [.5, 2., 8., 16.]:
        scores[f'adv_score_l2_eps_{eps}'] = tools.get_adv_score(adv_model, dataset, constraint='2', eps=eps, itrerations=7)
    
    # hessian
    criterion = torch.nn.CrossEntropyLoss()

    model = torch.nn.DataParallel(model)
    transform = transforms.ToTensor()
    loader = DataLoader(dataset, batch_size=64,
                        shuffle=True, num_workers=0, pin_memory=True)
    H = hessian(model, criterion, data=None, dataloader=loader, cuda=True)
    scores[f'top_eigenvalues'], _ = H.eigenvalues(top_n=1,maxIter=20)
    scores['trace'] = H.trace()
    scores['density'] = H.density()


    # save results
    model_id = re.findall(r'id-(\d+)/', model_filepath)[0]
    pd.DataFrame(scores, [0]).to_csv(f'results/{model_id}.csv')




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='/home/ubuntu/round2/models/id-00000000/model.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='/home/ubuntu/round2/models/id-00000000/example_data')

    args = parser.parse_args()
    trojan_detector(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)
