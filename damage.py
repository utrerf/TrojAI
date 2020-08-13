#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:18:28 2020

@author: ben
"""

import numpy as np
import skimage.io
import random
import torch
import warnings 
import pandas as pd


import torch.nn.functional as F
from torch.autograd import Variable, grad
import torch.nn.utils.prune as prune

from utils import test_acc, get_softmax

import copy

warnings.filterwarnings("ignore")


def damage(model, data, targets, N, n_batch):

    
    
    num_data = N*n_batch
    
    X_ori = torch.Tensor(num_data, 3, 224, 224)
    Y_test = torch.LongTensor(num_data)
            

    #print('Run IFGSM')
    for i in range(n_batch):
        X_ori[i*N:(i+1)*N] = data[i*N:(i+1)*N].float()
        Y_test[i*N:(i+1)*N] = torch.from_numpy(targets[i].flatten()).cuda()
                                   
    
    model_ori = copy.deepcopy(model)
    

    temp = []
    for i in range(12):
            
            for child in model.children():
              for layer in child.modules():
                #if(isinstance(layer,torch.nn.modules.conv.Conv2d) or isinstance(layer,torch.nn.modules.Linear)):
                
                if(isinstance(layer,torch.nn.modules.conv.Conv2d)):                        
                        
                        
                        np.random.seed(i*1000)
                        if np.min(layer.weight.data.shape) > 1:
                                
                            if np.random.binomial(1,0.7) == 1:
                                    # Damage
                                    sparse = np.random.binomial(1, 0.9, layer.weight.data.shape)
                                    layer.weight.data = layer.weight.data * torch.from_numpy(sparse).float().cuda()
           
                                    gamma = torch.abs(layer.weight.data).max()
                                    kappa = gamma * 0.4
                                    
                                    # Trimming                  
                                    layer.weight.data[layer.weight.data>kappa] = kappa
                                    layer.weight.data[layer.weight.data<-kappa] = -kappa                        
                                    
                                    
                                    # Pruning
                                    layer.weight.data[torch.abs(layer.weight.data)<gamma*0.09] = 0.0                        
                        
                
            result_acc, result_ent = test_acc(X_ori, Y_test, model, num_data, N)
            temp.append(result_acc)
            model = copy.deepcopy(model_ori)
        
    print(np.round(np.asarray(temp), 3))
    print(np.mean(np.asarray(temp)))
    
    score = np.mean(np.asarray(temp))


    if score <= 0.3:
        damage_prob = 1/(1+ np.exp( 10 * (score - 0.3)))
    elif score >= 0.75:
        damage_prob = 1/(1+ np.exp( 10 * (score - 0.75)))
    else:
        damage_prob = 0.5

    return damage_prob




def damage2(model, data, targets, N, n_batch):

    
    
    num_data = N*n_batch
    
    X_ori = torch.Tensor(num_data, 3, 224, 224)
    Y_test = torch.LongTensor(num_data)
            

    #print('Run IFGSM')
    for i in range(n_batch):
        X_ori[i*N:(i+1)*N] = data[i*N:(i+1)*N].float()
        Y_test[i*N:(i+1)*N] = torch.from_numpy(targets[i].flatten()).cuda()
                                   
    
    model_ori = copy.deepcopy(model)
    

    temp = []
    for i in range(12):
        
            k = 0
            for child in model.children():
                  for layer in child.modules():
                      
                      k+=1 
                      if k > 5:
                             
                            #if(isinstance(layer,torch.nn.modules.conv.Conv2d) or isinstance(layer,torch.nn.modules.Linear)):
                            if(isinstance(layer,torch.nn.modules.Linear)):                        
                                    
                                    np.random.seed(i*1000)
                                    if np.min(layer.weight.data.shape) > 1:
                                        layer.weight.data = layer.weight.data + torch.from_numpy(np.random.standard_normal(layer.weight.data.shape)*0.7 ).float().cuda() 
            
                            if(isinstance(layer,torch.nn.modules.Linear)):
            
                                    np.random.seed(i*1000)
                                    if np.min(layer.weight.data.shape) > 1:
                                        layer.weight.data = layer.weight.data + torch.from_numpy(np.random.standard_normal(layer.weight.data.shape)*0.09 ).float().cuda() 
            

                
            result_acc, result_ent = test_acc(X_ori, Y_test, model, num_data, N)
            temp.append(result_acc)
            model = copy.deepcopy(model_ori)
        
    print(np.round(np.asarray(temp), 3))
    print(np.mean(np.asarray(temp)))
    score = np.mean(np.asarray(temp))


    if score <= 0.35:
        damage_prob = 1/(1+ np.exp( 12 * (score - 0.35)))
    elif score >= 0.5:
        damage_prob = 1/(1+ np.exp( 15 * (score - 0.5)))
    else:
        damage_prob = 0.5

    return damage_prob