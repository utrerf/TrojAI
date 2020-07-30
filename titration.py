#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:30:55 2020

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

from utils import test_acc

warnings.filterwarnings("ignore")


def get_softmax_titration(model, sigma, N, n_batch, dim, channels, data):
    np.random.seed(123)
    model = model.cuda()

    softmax_activations_collect = []
    pred_collect =  []

    for i in range(n_batch):
        
        np.random.seed(i)
        noise = np.random.standard_normal((N,channels,dim,dim))
        
        with torch.no_grad():
            
            tit_noise = torch.from_numpy((noise * sigma)).float().cuda() 
            data_batch = data[i*N:(i+1)*N] + tit_noise
                        
            output = model(data_batch)
                    
            softmax_activations = F.softmax(output, dim=1)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                   
            
        softmax_activations_collect.append(softmax_activations.data.cpu().numpy())
        pred_collect.append(pred.data.cpu().numpy())
    
    
    pred_collect = np.asarray(pred_collect)
    softmax_activations_collect = np.asarray(softmax_activations_collect)
    
    return softmax_activations_collect, pred_collect 



def tscore(softmax_activation, preds, targets, gamma=0.95):

    confidence_id = 0
    conf = 0
    for i in range(softmax_activation.shape[0]):
        
        soft_vals = softmax_activation[i]
        idx = np.where(preds[i] != targets[i])[0]
        soft_vals = soft_vals[idx]
        
        confidence_id = np.where(np.max(np.abs(soft_vals), axis=1) > gamma)[0]
        conf += len(confidence_id)
    
    tscore = conf / (preds.shape[0]*preds.shape[1])       
    return tscore














