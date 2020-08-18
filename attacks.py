#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:03:26 2020

@author: ben
"""

import numpy as np
import torch
import warnings 
import pandas as pd


import torch.nn.functional as F

from utils import test_acc, get_softmax

from attack_tools import *

from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")












def attack(model, data, targets, N, n_batch, eps, iters, df_iter, p=1):
    
        #torch.manual_seed(123456)
        #torch.cuda.manual_seed(123456)
    
        c = np.max(targets.cpu().numpy().flatten())
        num_data = N*n_batch
        X_ori = torch.Tensor(num_data, 3, 224, 224)
        X_fgsm = torch.Tensor(num_data, 3, 224, 224)
        
        for i in range(n_batch):
            X_ori[i*N:(i+1)*N] = data[i*N:(i+1)*N].float()           
    

        eps = 0.0005
        iters = 10
        df_iter = 3
        
        for i in range(n_batch):
            X_fgsm[i*N:(i+1)*N] = fgsm_iter(model, data[i*N:(i+1)*N].float(), targets[i*N:(i+1)*N].flatten().long().cuda(), eps, iterations=iters)
                                       
            
#        cross = torch.nn.CrossEntropyLoss(reduction = 'sum')
#        score1 = 0
#        for i in range(n_batch):
#                with torch.no_grad():
#                    data_batch = X_fgsm[i*N:(i+1)*N].float().cuda()
#                    output = model(data_batch)                            
#                    score1 += cross(output.float().cuda(), targets[i*N:(i+1)*N].long().cuda()).item()       
#        score1 /= (N*n_batch)  


        preds = torch.Tensor(num_data,1)
        for i in range(n_batch):
                with torch.no_grad():
                    data_batch = X_fgsm[i*N:(i+1)*N].float().cuda()
                    output = model(data_batch)       
                    preds[i*N:(i+1)*N] = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                           
        score1 = f1_score(targets.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='macro')       
        
                    
        
        
        
     
        _, absnoise1,  _ = distance(X_fgsm, X_ori, norm=2)
        print('Attack score 1', score1)
        print('Attack noise 1', absnoise1.item())
    
    
        for i in range(n_batch):
            X_fgsm[i*N:(i+1)*N], _ = deep_fool_iter(model, X_fgsm[i*N:(i+1)*N].float(), targets[i*N:(i+1)*N].flatten().long().cuda(), c=c, p=2, iterations=df_iter)
    
    
#        cross = torch.nn.CrossEntropyLoss(reduction = 'sum')
#        score2 = 0
#        for i in range(n_batch):
#                with torch.no_grad():
#                    data_batch = X_fgsm[i*N:(i+1)*N].float().cuda()
#                    output = model(data_batch)        
#                    score2 += cross(output.float().cuda(), targets[i*N:(i+1)*N].long().cuda()).item()       
#        score2 /= (N*n_batch)  


        preds = torch.Tensor(num_data,1)
        for i in range(n_batch):
                with torch.no_grad():
                    data_batch = X_fgsm[i*N:(i+1)*N].float().cuda()
                    output = model(data_batch)       
                    preds[i*N:(i+1)*N] = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                           
        score2 = f1_score(targets.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='macro')  
    
        _, absnoise2,  _ = distance(X_fgsm, X_ori, norm=2)
    
        print('Attack score 2', score2)
        print('Attack noise 2', absnoise2.item())

        return score1, score2, absnoise1.item(), absnoise2.item()