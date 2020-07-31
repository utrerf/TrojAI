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

from utils import test_acc, get_softmax

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




def titration(model, N, n_batch):

        N = 400
        n_batch = 1
        # copy data back
        np.random.seed(0)
        img = np.random.standard_normal((N, 3, 224, 224)) * 0.05
        for i in range(img.shape[0]):
            img[i] = img[i] - np.min(img[i])
            img[i] = img[i] / np.max(img[i])    
        
        data_art = torch.from_numpy(img).float().cuda()
        
        
        true_out, targets_art = get_softmax(model, N=N, n_batch=n_batch, dim=224, channels=3, data=data_art)
        
        
    
        tscores = []
        sigs = [0.0, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 10, 20, 50, 100, 200]
        sigs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    
        for sigma in sigs:
            soft_titration, preds_titration = get_softmax_titration(model, sigma, N=N, n_batch=n_batch, dim=224, channels=3, data=data_art)
            tscores.append(tscore(soft_titration, preds_titration, targets_art, gamma=0.95))
        
        print(tscores)
        
        
        #    ================================================
        #     Do some Damage
        #    ================================================
        
    
        k = 0   
        for child in model.children():
          for layer in child.modules():
            if(isinstance(layer,torch.nn.modules.conv.Conv2d) or isinstance(layer,torch.nn.modules.Linear)):
                    
                if k < 15:
                    gamma = torch.abs(layer.weight.data).max()
                    kappa = gamma * 0.5
                    
                    # Trimming                  
                    layer.weight.data[layer.weight.data>kappa] = kappa
                    layer.weight.data[layer.weight.data<-kappa] = -kappa
                        
                    # Damage
                    #np.random.seed(123)
                    #sparse = np.random.binomial(1, 0.95, layer.weight.data.shape)
                    #layer.weight.data = layer.weight.data * torch.from_numpy(sparse).float().cuda()
                        
                    # Pruning
                    layer.weight.data[torch.abs(layer.weight.data)<gamma*0.1] = 0.0
                    k += 1
                        
        
        
        tscores = []
        sigs = [0.0, 0.5, 1.0, 1.5, 2.0, 4.0, 6.0, 10, 20, 50, 100, 200]
        sigs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]
    
        for sigma in sigs:
            soft_titration, preds_titration = get_softmax_titration(model, sigma, N=N, n_batch=n_batch, dim=224, channels=3, data=data_art)
            tscores.append(tscore(soft_titration, preds_titration, targets_art, gamma=0.95))
        
        print(tscores)    

        titration_prob = 0.5

        return titration_prob








