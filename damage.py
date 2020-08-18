#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:18:28 2020

@author: ben
"""

import numpy as np
import skimage.io
import torch
import warnings 
import pandas as pd


import torch.nn.functional as F
from torch.autograd import Variable, grad
import torch.nn.utils.prune as prune

from utils import test_acc, get_softmax, test_cross

import copy

from sklearn.metrics import f1_score


warnings.filterwarnings("ignore")


def damage(model, data, targets, N, n_batch, trimm=0.55, prune=0.05):

 
    model_ori = copy.deepcopy(model)
    temp = []

#    cross = torch.nn.CrossEntropyLoss(reduction = 'sum')
#    score1 = 0
#    for i in range(n_batch):
#            with torch.no_grad():
#                data_batch = data[i*N:(i+1)*N].float().cuda()
#                output = model(data_batch)        
#                score1 += cross(output.float().cuda(), targets[i*N:(i+1)*N].long().cuda()).item()       
#    score1 /= (N*n_batch)  

    preds = torch.Tensor(N*n_batch,1)
    for i in range(n_batch):
                with torch.no_grad():
                    data_batch = data[i*N:(i+1)*N].float().cuda()
                    output = model(data_batch)       
                    preds[i*N:(i+1)*N] = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                           
    score1 = f1_score(targets.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='macro')  

    
    for i in range(1):
        
            k = 0
            for child in model.children():
                  for layer in child.modules():
                      
                    if(isinstance(layer,torch.nn.modules.conv.Conv2d)):                        
                            
                            
                            np.random.seed(i*1000)
                            if np.min(layer.weight.data.shape) > 0:
                                    
                                #if np.random.binomial(1,0.7) == 1:
                                        # Damage
                                        #sparse = np.random.binomial(1, 0.9, layer.weight.data.shape)
                                        #layer.weight.data = layer.weight.data * torch.from_numpy(sparse).float().cuda()
               
                                        gamma = torch.abs(layer.weight.data).max()
                                        kappa = gamma * trimm
                                        
                                        # Trimming                  
                                        layer.weight.data[layer.weight.data>kappa] = kappa
                                        layer.weight.data[layer.weight.data<-kappa] = -kappa                        
                                        
                                        
                                        # Pruning
                                        layer.weight.data[torch.abs(layer.weight.data)<gamma*prune] = 0.0   
                
            preds = torch.Tensor(N*n_batch,1)
            for i in range(n_batch):
                        with torch.no_grad():
                            data_batch = data[i*N:(i+1)*N].float().cuda()
                            output = model(data_batch)       
                            preds[i*N:(i+1)*N] = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                                   
            score2 = f1_score(targets.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='macro')               
            
            
            temp.append(np.abs(score1-score2))
            
            
            model = copy.deepcopy(model_ori)
        
    #print(np.round(np.asarray(temp), 3))
    print('Damage score: ', np.mean(np.asarray(temp)))
    score = np.mean(np.asarray(temp))

    return score