import skimage.io
import re
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from sklearn.metrics import f1_score
from torchvision import transforms
from PIL import Image
from robustness.datasets import DataSet 
import sys
sys.path.append('PyHessian')
from pyhessian import hessian


def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class TrojAIDataset(Dataset):
    # Makes the dataset from examples_dirpath
    def __init__(self, examples_dirpath, transform=None):
        self.examples_dirpath = examples_dirpath
        self.transform = transform
        self.x, self.y = [], []
        center_crop_transform = transforms.CenterCrop(224)
        for image_filename in os.listdir(examples_dirpath):
            img = Image.open(os.path.join(examples_dirpath, image_filename))
            img = img.convert('RGB')
            img = center_crop_transform(img)
            img = transforms.ToTensor()(img)
            self.x.append(img)
        
            label = int(re.findall(r'class_(\d+)_', image_filename)[0])
            self.y.append(label)
        self.y = torch.LongTensor(self.y)
        self.default_transform = transforms.ToTensor()

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        if self.transform: 
            x = self.transform(x)
        #else:
        #    x = self.default_transform(x)
        return x, y

    def __len__(self):
        return len(self.x)


def get_model_info(model): 
    model_info = {}
    total_parameters = 0
    for parameter in model.parameters():
        num_parameters = 1
        for dim in parameter.size():
            num_parameters *= dim
        total_parameters += num_parameters
    model_info['total_parameters'] = total_parameters
    model_info['num_classes'] = len(parameter)
    return model_info


def transform_scores(model, dataset, transform, scores, score,
                        num_iterations=5, num_workers=0, batch_size=120):
    dataset.transform = transform
    loader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=True, num_workers=num_workers, pin_memory=False)
    return transform_scores_helper(model, loader, num_iterations, scores, score)


def transform_scores_helper(model, loader, num_iterations, scores, score):
    # compute the mean f1 score over num_iterations
    f1_sum, avg_abs_dev_sum, sum_std_pred_bins = 0, 0, 0
    for iteration in range(num_iterations):
        n = len(loader.dataset)
        preds, targets = np.zeros(n), np.zeros(n)
        current_idx = 0
        for inp, target in loader:
            with torch.no_grad():
                inp = inp.cuda()
                output = model(inp)
                m = len(target)
                argmax =  torch.argmax(output, dim=1).type(torch.uint8).cpu().numpy().flatten()
                preds[current_idx : current_idx+m] += argmax
                targets[current_idx : current_idx+m] += target.numpy().flatten()
                current_idx += m
        it_scores = get_scores(targets, preds, n)
        f1_sum += it_scores['f1']
        avg_abs_dev_sum += it_scores['aad']
        sum_std_pred_bins += it_scores['aad_std']

    f1, aad, aad_std = f1_sum/num_iterations, avg_abs_dev_sum/num_iterations, sum_std_pred_bins/num_iterations
    scores[f'f1_{score}'], scores[f'aad_{score}'], scores[f'aad_std_{score}'] = f1, aad, aad_std 
    return scores


def get_scores(targets, preds, n):
    scores = {}
    scores['f1'] = f1_score(targets, preds, average='macro')
    
    num_classes = len(np.unique(targets))
    pred_bins = np.bincount(preds.astype(int), minlength=num_classes)/n
    target_bins = np.bincount(targets.astype(int), minlength=num_classes)/n
    scores['aad']= np.mean(np.abs(pred_bins - target_bins))
    scores['aad_std'] = np.std(pred_bins)

    return scores


class MadryDataset(DataSet):
    def __init__(self, data_path, **kwargs):
        ds_name = 'madry_dataset'
        ds_kwargs = {
            'num_classes': kwargs['num_classes'],
            'mean': torch.tensor([0., 0., 0.]).cuda(),
            'std': torch.tensor([1., 1., 1.]).cuda(),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': None,
            'transform_test': None,
            }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(MadryDataset, self).__init__(ds_name,
                                           data_path, **ds_kwargs)

class CustomAdvModel(torch.nn.Module):
    def __init__(self, model):
        super(CustomAdvModel, self).__init__()
        self.model = model
    def forward(self, inp, with_latent=False,
            fake_relu=False, no_relu=False):
        output = self.model(inp)
        return output


def adv_scores(adv_model, dataset, scores, score, constraint='inf', eps=0.04, iterations=3, 
                  transform=None, batch_size=60, num_workers=0,
                  compute_top_eigenvalue=False):
    if constraint in ['2', 'inf']:
        attack_kwargs = {
                'constraint': constraint, # L-inf PGD
                'eps': eps, # Epsilon constraint (L-inf norm)
                'step_size': 2.*(eps/iterations), # Learning rate for PGD
                'iterations': iterations, # Number of PGD steps
                'targeted': False, # Targeted attack
                'custom_loss': None, # Use default cross-entropy loss
                'random_start': True
                }
    else:
        # TODO: Add chop_func attack_kwargs
        attack_kwargs = {
                }
    dataset.transform = transform
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=num_workers, pin_memory=False)
    return adv_scores_helper(adv_model, loader, attack_kwargs, 
                               compute_top_eigenvalue, scores, score)


def adv_scores_helper(adv_model, loader, attack_kwargs, compute_top_eigenvalue, scores, score):
    n = len(loader.dataset) 
    preds, targets   = np.zeros(len(loader.dataset)), np.zeros(len(loader.dataset))
    dataset_size = [n] + list(loader.dataset[0][0].shape)
    adv_images = torch.empty(dataset_size, device='cpu', requires_grad=False)
    current_idx = 0
    for inp, target in loader:
        if attack_kwargs['constraint'] in ['2', 'inf']:
            output, im_adv = adv_model(inp.cuda(), target.cuda(), make_adv=True, **attack_kwargs)
        else: 
            # TODO: feel free to change inputs to the function, but please keep outputs fixed
            #       adv_model.model takes the model out of the madry wrapper class
            output, im_adv = chop_func(adv_model.model, inp.cuda(), target.cuda(), **attack_kwargs)
            
        with torch.no_grad():
            m = len(target)
            argmax =  torch.argmax(output, dim=1).cpu().numpy().flatten()
            preds[current_idx : current_idx+m] += argmax
            targets[current_idx : current_idx+m] += target.numpy().flatten()
 
            if compute_top_eigenvalue:
                adv_images[current_idx : current_idx+m] = im_adv.detach().clone().cpu()
            
            current_idx += m
        del inp, target, output, im_adv 
   
    adv_dataset = None
    if compute_top_eigenvalue:
        adv_dataset = torch.utils.data.TensorDataset(adv_images, torch.from_numpy(targets))

    it_scores = get_scores(targets, preds, n)
    for key, val in it_scores.items():
        scores[f'{key}_{score}'] = val
    
    return scores, adv_dataset


def chop_func(model, inp, target, **attack_kwargs):
    # TODO: fill out
    return output, im_adv


def compute_top_eigenvalue(model, dataset, scores, score, batch_size=60, num_workers=0, 
                                       criterion=torch.nn.CrossEntropyLoss(), max_iter=10):
    loader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=True, num_workers=num_workers, pin_memory=False)
    H = hessian(model, criterion, data=None, dataloader=loader, cuda=True)
    top_eigenvalue, _ = H.eigenvalues(top_n=1,maxIter=max_iter) 
    del _, H
    torch.cuda.empty_cache()
    scores[f'top_eig_{score}'] = top_eigenvalue
    return scores


def compute_grad_l2_norm(model, dataset, scores, score, batch_size=40,num_workers=0,
                                              criterion=torch.nn.CrossEntropyLoss()):
    loader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=True, num_workers=num_workers, pin_memory=False)
    for i, (inp, target) in enumerate(loader):
        loss = criterion(model(inp.cuda()),target.cuda())
        loss.backward()
        if i == 0:
            v = [p.grad.data.flatten().cpu() for p in model.parameters()]
        else:
            v = [v1 + p.grad.data.flatten().cpu() for v1,p in zip(v,model.parameters())]
    scores[f'grad_l2_norm_{score}'] = torch.cat(v).norm(2).item()
    return scores


