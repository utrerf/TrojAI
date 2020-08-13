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


def spectral(model, data, targets, N, n_batch, k=2):

    
    num_data = N*n_batch
    
    X_ori = torch.Tensor(num_data, 3, 224, 224)
    Y_test = torch.LongTensor(num_data)
    for i in range(n_batch):
        X_ori[i*N:(i+1)*N] = data[i*N:(i+1)*N].float()
        Y_test[i*N:(i+1)*N] = torch.from_numpy(targets[i].flatten()).cuda()
                                   
    
     
    for child in model.children():
        for layer in child.modules():
                #if(isinstance(layer,torch.nn.modules.conv.Conv2d) or isinstance(layer,torch.nn.modules.Linear)):
                if(isinstance(layer,torch.nn.modules.Linear)):                        
                        

                    try:
                        u, s, v = torch.svd(layer.weight.data)
                    
                        fnorm = np.linalg.norm(layer.weight.data.cpu().numpy())
                        srank = fnorm**2 / s[0]**2
                        print('Stable rank: ', srank.item())
                        srank = int(np.ceil(srank.item()+k))
                        srank = np.minimum(srank, np.min(layer.weight.data.shape))
                        
                        if srank > (np.min(layer.weight.data.shape)):
                            srank -= 2
                    
                        layer.weight.data = torch.mm(torch.mm(u[:,0:srank], torch.diag(s[0:srank])), v[:,0:srank].t())
                        
                    except RuntimeError:
                        print('LinAlg error')
                    

    result_acc, result_ent = test_acc(X_ori, Y_test, model, num_data, N)
        

    if result_acc <= 0.25:
        damage_prob = 1/(1+ np.exp( 8 * (result_acc - 0.25)))
    elif result_acc >= 0.8:
        damage_prob = 1/(1+ np.exp( 8 * (result_acc - 0.8)))
    else:
        damage_prob = 0.5

    return damage_prob