import torch
import argparse
import tools
import numpy as np
import custom_transforms
from robustness import attacker

def trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, example_img_format='png'):
    tools.set_seeds(123)
    dataset = tools.TrojAIDataset(examples_dirpath)
    
    model = torch.load(model_filepath) 
    model_info = tools.get_model_info(model)
    model.cuda().eval()
    
    scores = {}
    # titration
    for noise_level in [0., .1, .2, .4, .8]:
        transform = custom_transforms.TITRATION_TRANSFORM(noise_level)
        scores[f't_score_noise_level_{noise_level}'] = tools.get_transform_score(model, dataset, transform)
    # erase
    for erase_probability in [0., .1, .25, .5, .75, .9]:
        transform = custom_transforms.ERASE_TRANSFORM(erase_probability) 
        scores[f't_score_erase_probability_{noise_level}'] = tools.get_transform_score(model, dataset, transform)
    # adversarial
    adv_dataset = tools.MadryDataset(None, num_classes=model_info['num_classes'])
    adv_model = attacker.AttackerModel(model, adv_dataset)
    for eps in [0, .01, .02, .04, .08, .16]:
        scores[f'adv_score_eps_{eps}'] = tools.get_adv_score(adv_model, dataset, eps=eps)
    print(scores)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='/scratch/erichson/TrojAI/id-00000000/model.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='/scratch/erichson/TrojAI/id-00000000/clean_example_data')

    args = parser.parse_args()
    trojan_detector(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)
