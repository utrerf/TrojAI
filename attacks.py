#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:03:26 2020

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

from attack_tools import *


warnings.filterwarnings("ignore")












def attack(model, data, targets, N, n_batch, eps, iters, df_iter, p=1):
    
    
    eps = 0.0001
    iters = 80

    eps = 0.0005
    iters = 20    
    
    df_iter = 2 
      
    
    c = np.max(targets.flatten())
    
    
    num_data = N*n_batch
    
    X_ori = torch.Tensor(num_data, 3, 224, 224)
    X_fgsm = torch.Tensor(num_data, 3, 224, 224)
    Y_test = torch.LongTensor(num_data)

    _, targets = get_softmax(model, N=N, n_batch=n_batch, dim=224, channels=3, data=data)
           

    #print('Run IFGSM')
    for i in range(n_batch):
        X_ori[i*N:(i+1)*N] = data[i*N:(i+1)*N].float()
        Y_test[i*N:(i+1)*N] = torch.from_numpy(targets[i].flatten()).cuda()
        X_fgsm[i*N:(i+1)*N], iters_fgsm = fgsm_adaptive_iter(model, data[i*N:(i+1)*N].float(), torch.from_numpy(targets[i].flatten()).cuda(), eps, iterations=iters)
                                   
    

    #print(iters_fgsm)
    #result_acc, result_ent = test_acc(X_ori, Y_test, model, num_data, N)
    #print(result_acc)
    
    result_acc, result_ent = test_acc(X_fgsm, Y_test, model, num_data, N)        
    print(result_acc)
 
    #_, result_dis_abs_pgd,  result_large_pgd = distance(X_fgsm, X_ori, norm=2)
    #_, result_dis_abs_pgd_inf,  result_large= distance(X_fgsm, X_ori, norm=1)
    
    #pgd_prob =  1/(1+ np.exp( 20 * (result_acc - 0.56)))



    #------------
    # Deep Fool
    #------------
    for i in range(n_batch):
        X_fgsm[i*N:(i+1)*N], _ = deep_fool_iter(model, X_fgsm[i*N:(i+1)*N].float(), torch.from_numpy(targets[i].flatten()).cuda(), c=c, p=1, iterations=df_iter)

    result_acc_df, result_ent = test_acc(X_fgsm, Y_test, model, num_data, N)        
    print(result_acc_df)     

    df_prob =  1/(1+ np.exp( 10 * (result_acc_df - 0.45)))

    if result_acc_df <= 0.40:
        df_prob =  1/(1+ np.exp( 8 * (result_acc_df - 0.40)))
    elif result_acc_df >= 0.5:
        df_prob =  1/(1+ np.exp( 8 * (result_acc_df - 0.5)))
    else:
        df_prob = 0.5


    #_, result_dis_abs_df,  result_large= distance(X_fgsm, X_ori, norm=2)
    #print('PGD Abs. Noise (2-norm): ', np.round(result_dis_abs_pgd, 2))
    #print('DF Abs. Noise (2-norm): ', np.round(result_dis_abs_df, 2))

    #_, result_dis_abs_df_inf,  result_large= distance(X_fgsm, X_ori, norm=1)
    #print('PGD Abs. Noise (inf-norm): ', np.round(result_dis_abs_pgd_inf, 2))
    #print('DF Abs. Noise (inf-norm): ', np.round(result_dis_abs_df_inf, 2))
    
    #print('DF Probability: {}'.format(pgd_prob))
    #print('DF Probability: {}'.format(df_prob))    

    #if result_dis_abs_pgd < 2.3 and df_prob > 0.5:
    #    df_prob = 0.5
        
    #if result_dis_abs_df > 10 and df_prob > 0.5:
    #    df_prob = 0.5


    return df_prob, result_acc_df