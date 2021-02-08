import re
import os
import skimage.io
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import entropy
import torch
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from robustness.datasets import DataSet 
import chop


def set_seeds(seed):
    """ Sets the seeds for numpy, torch, and cuda """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class TrojAIDataset(Dataset):
    """ Class used to make the dataset from examples_dirpath """
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
        return x, y

    def __len__(self):
        return len(self.x)


def get_model_info(model): 
    """ Return total parameters and number of classes in model """
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
    """ Obtains features by appliying a transform on the inputs """
    dataset.transform = transform
    loader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=True, num_workers=num_workers, pin_memory=False)
    return transform_scores_helper(model, loader, num_iterations, scores, score)


def transform_scores_helper(model, loader, num_iterations, scores, score):
    """ computes the scores averaged over various iterations """
    it_scores_sum = {}
    for iteration in range(num_iterations):
        n = len(loader.dataset)
        preds, targets = np.zeros(n), np.zeros(n)
        current_idx = 0
        for inp, target in loader:
            with torch.no_grad():
                output = model(inp.cuda())
                argmax =  torch.argmax(output, dim=1).type(torch.uint8).cpu().numpy().flatten()
                m = len(target)
                preds[current_idx : current_idx+m] += argmax
                targets[current_idx : current_idx+m] += target.numpy().flatten()
                current_idx += m
        it_scores = get_scores(targets, preds, n)
        if iteration == 0:
            it_scores_sum = it_scores
        else: 
            for k in it_scores.keys():
                it_scores_sum[k] += it_scores[k]
    
    return {k:v/num_iterations for k, v in it_scores_sum.items()}


def get_scores(targets, preds, n):
    """ computes features that will later be used to train a classifier for posioned models
            targets: n-dimensional vector of integers representing the class of an input
            preds: n-dimensional vector if integers representing the argmax prediction of the model
            n: number of inputs
    """
    scores = {}
    scores['f1'] = f1_score(targets, preds, average='macro')
    
    num_classes = len(np.unique(targets))
    pred_bins = np.bincount(preds.astype(int), minlength=num_classes)/n
    target_bins = np.bincount(targets.astype(int), minlength=num_classes)/n
    scores['aad']= np.mean(np.abs(pred_bins - target_bins))
    scores['aad_std'] = np.std(pred_bins)
    scores['entropy'] = entropy(pred_bins)
    scores['kl'] = entropy(pred_bins, target_bins)

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
                  transform=None, batch_size=20, num_workers=0, adversary_alg=None):
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
        attack_kwargs = {
            'constraint': constraint,
            'eps': eps,
            'criterion': torch.nn.CrossEntropyLoss(reduction='none'),
            'adversary': chop.Adversary(adversary_alg),
            'iterations': iterations,
                }

    dataset.transform = transform
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=num_workers, pin_memory=False)
    return adv_scores_helper(adv_model, loader, attack_kwargs, scores, score)


def adv_scores_helper(adv_model, loader, attack_kwargs, scores, score):
    n = len(loader.dataset) 
    preds, targets   = np.zeros(len(loader.dataset)), np.zeros(len(loader.dataset))
    dataset_size = [n] + list(loader.dataset[0][0].shape)
    triggers = torch.empty(dataset_size, device='cpu', requires_grad=False)
    
    current_idx = 0
    for inp, target in loader:
        if attack_kwargs['constraint'] in ['2', 'inf']:
            output, im_adv = adv_model(inp.cuda(), target.cuda(), make_adv=True, **attack_kwargs)
        else: 
            output, im_adv = chop_func(adv_model.model, inp.cuda(), target.cuda(), **attack_kwargs)
            
        with torch.no_grad():
            m = len(target)
            argmax =  torch.argmax(output, dim=1).cpu().numpy().flatten()
            preds[current_idx : current_idx+m] += argmax
            targets[current_idx : current_idx+m] += target.numpy().flatten()
            trigger = im_adv - inp.cuda() 
            triggers[current_idx : current_idx+m] = trigger.detach().clone().cpu()
            
            current_idx += m
        del inp, target, output, im_adv 
     
    triggers_dataset = torch.utils.data.TensorDataset(triggers, torch.from_numpy(preds).type(torch.long))
    it_scores = get_scores(targets, preds, n)
    for key, val in it_scores.items():
        scores[f'{key}_{score}'] = val
    
    return scores, triggers_dataset


def chop_func(model, inp, target, **attack_kwargs):
    
    if attack_kwargs['constraint'] == '1':
        constraint = chop.constraints.L1ball(attack_kwargs['eps'])
    elif attack_kwargs['constraint'] in ['groupLasso', 'groupL1']:
        groups = chop.image.group_patches(x_patch_size=8, y_patch_size=8,
                                          x_image_size=inp.size(-2),
                                          y_image_size=inp.size(-1)
                                          )
        constraint = chop.constraints.GroupL1Ball(attack_kwargs['eps'] * len(groups), groups)
    # aliases for the sum of eigenvalue ball
    elif attack_kwargs['constraint'] in ['nuclearnorm', 'tracenorm']:
        constraint = chop.constraints.NuclearNormBall(attack_kwargs['eps'])

    adversary = attack_kwargs['adversary']

    # define projection on image space (0, 1) box constraint
    def image_constraint_prox(delta, step_size=None):
        adv_img = torch.clamp(inp + delta, 0, 1)
        delta = adv_img - inp 
        return delta

    if adversary.method in [chop.optim.minimize_frank_wolfe,
                            chop.optim.minimize_pgd_madry]:
        attack_kwargs['lmo'] = constraint.lmo
        
    elif adversary.method == chop.optim.minimize_three_split:
        attack_kwargs['prox1'] = constraint.prox
        attack_kwargs['prox2'] = image_constraint_prox

    _, delta = adversary.perturb(inp, target, model,
                                 max_iter=attack_kwargs['iterations'], **attack_kwargs)

    delta = image_constraint_prox(delta)
    im_adv = inp + delta
    output = model(im_adv)
    
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


def compute_density(model, dataset, scores, score, batch_size=60, num_workers=0, 
                                       criterion=torch.nn.CrossEntropyLoss(), max_iter=10):
    loader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=True, num_workers=num_workers, pin_memory=False)
    H = hessian(model, criterion, data=None, dataloader=loader, cuda=True)
    density_eigen, density_weight = H.density(iter=max_iter) 
    print(f'density_eigen:  {density_eigen}')
    print(f'density_weight: {density_weight}')
    del H
    torch.cuda.empty_cache()
    scores[f'density_eigen_{score}'] = density_eigen
    scores[f'density_weight_{score}'] = density_weight
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


class Triggered_Dataset(Dataset):
    def __init__(self, clean_dataset, trigger, trigger_class, transform=None):
        self.clean_dataset = clean_dataset
        self.trigger = trigger
        self.y = trigger_class
        self.transform = transform

    def __getitem__(self, index):
        x, _ = self.clean_dataset[index]
        x = torch.clamp(x + self.trigger, 0 , 1)
        if self.transform: 
            x = self.transform(x)
        return x, self.y

    def __len__(self):
        return len(self.clean_dataset)


def basic_artificial_trigger_success(model, dataset, adv_datasets, scores,
                                     batch_size=20, num_workers=0):
    for score, adv_dataset in adv_datasets.items():
        max_trigger_success = 0
        for trigger, trigger_class in adv_dataset:
            trig_ds = Triggered_Dataset(dataset, trigger, trigger_class)
            trig_loader = DataLoader(trig_ds, batch_size=batch_size, 
                                     shuffle=True, num_workers=num_workers, pin_memory=False)
            acc_sum = 0
            for inp, target in trig_loader:
                with torch.no_grad():
                    output = model(inp.cuda())
                    argmax =  torch.argmax(output, dim=1).type(torch.uint8).cpu().numpy().flatten()
                    acc_sum += np.sum(argmax == target.numpy().flatten())
            acc = acc_sum/len(trig_loader.dataset)
            if acc > max_trigger_success:
                max_trigger_success = acc
        
        scores[score+'_artficial_trigger_acc'] = max_trigger_success
    return scores


def set_madry_fgsm_attack(constraint, eps, iterations, target):
    attack_kwargs = {
        'constraint': constraint,         # ell_p norm constraint
        'eps': eps,                       # Epsilon constraint
        'step_size': 2.*(eps/iterations), # Learning rate
        'iterations': 1,                  # Number of PGD steps
        'targeted': target,               # Targeted attack
        'custom_loss': None,              # Use default cross-entropy loss
        'random_start': False
        }
    return attack_kwargs
