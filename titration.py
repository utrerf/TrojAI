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

from attack_tools import *


warnings.filterwarnings("ignore")


def get_softmax_titration(model, sigma, N, n_batch, dim, channels, data):
    np.random.seed(10000)
    model = model.cuda()

    softmax_activations_collect = []
    pred_collect =  []

    for i in range(n_batch):
        
        np.random.seed(i)
        noise = np.random.standard_normal((N,channels,dim,dim))
        
        with torch.no_grad():
            
            tit_noise = torch.from_numpy((noise * sigma)).float().cuda() 
            data_batch = data[i*N:(i+1)*N].float().cuda()  + tit_noise.float().cuda() 
                        
            output = model(data_batch)
                    
            softmax_activations = F.softmax(output, dim=1)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                   
            
        softmax_activations_collect.append(softmax_activations.data.cpu().numpy())
        pred_collect.append(pred.data.cpu().numpy())
    
    
    pred_collect = np.asarray(pred_collect)
    softmax_activations_collect = np.asarray(softmax_activations_collect)
    
    return softmax_activations_collect, pred_collect 



def tscore(softmax_activation, preds, targets, gamma=0.9):

    confidence_id = 0
    conf = 0
    count = 0
    for i in range(softmax_activation.shape[0]):
        
        soft_vals = softmax_activation[i]
        idx = np.where(preds[i] != targets[i])[0]
        soft_vals = soft_vals[idx]
        
        confidence_id = np.where(np.max(np.abs(soft_vals), axis=1) > gamma)[0]
        conf += len(confidence_id)
        count += len(idx)
    
    #print(preds.shape[0]*preds.shape[1])
    tscore = conf / (count+1)       
    return tscore, count




def titration(model, N, n_batch):

        N = 500
        n_batch = 1
        num_data = N*n_batch
        
        # copy data back
        np.random.seed(10000)
        img = np.random.standard_normal((num_data, 3, 256, 256)) * 0.05
        #img2 = np.random.uniform(0,1,(num_data, 3, 256, 256))
        img = img #* img2
        
        for i in range(img.shape[0]):
            img[i] = img[i] - np.min(img[i])
            img[i] = img[i] / np.max(img[i])    
        
        
        data_art = torch.from_numpy(img).float().cuda()
        true_out, targets_art = get_softmax(model, N=N, n_batch=n_batch, dim=256, channels=3, data=data_art)
        c = np.max(targets_art.flatten())


        tscores = []
        sigs = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10]
        sigs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
        sigs = [0.2]
    
        for sigma in sigs:
            soft_titration, preds_titration = get_softmax_titration(model, sigma, N=N, n_batch=n_batch, dim=256, channels=3, data=data_art)
            ttemp, counttemp = tscore(soft_titration, preds_titration, targets_art, gamma=0.8)
            tscores.append(ttemp)
        
        print('titration: ', counttemp)
        
        
        titration_prob = 1 - 1/(1+ np.exp( 0.1 * (counttemp - 425)))

        if counttemp <= 100:
            titration_prob = 1 - 1/(1+ np.exp( 0.2 * (counttemp - 100)))
        elif counttemp >= 450:
            titration_prob = 1 - 1/(1+ np.exp( 0.2 * (counttemp - 450)))
        else:
            titration_prob = 0.5

        return titration_prob







def titration_real(model, data, targets, N, n_batch,):

        num_data = N*n_batch
                
        X_ori = torch.Tensor(num_data, 3, 224, 224)
        Y_test = torch.LongTensor(num_data)
                

        
        #print('Run IFGSM')
        for i in range(n_batch):
            X_ori[i*N:(i+1)*N] = data[i*N:(i+1)*N].float().cuda()
            Y_test[i*N:(i+1)*N] = torch.from_numpy(targets[i].flatten()).cuda()                                      
        
        _, Y_test = get_softmax(model, N=N, n_batch=n_batch, dim=224, channels=3, data=X_ori)
        
        
        for i in range(X_ori.shape[0]):
            for j in range(3):
                temp = X_ori[i,j, :, :].cpu().numpy()
                X_ori[i,j,:,:] = torch.from_numpy(temp.T).float().cuda()


        #soft_titration, preds_titration = get_softmax(model, N=N, n_batch=n_batch, dim=224, channels=3, data=X_ori)
        soft_titration, preds_titration = get_softmax_titration(model, 0.2, N=N, n_batch=n_batch, dim=224, channels=3, data=X_ori)

        ttemp, counttemp = tscore(soft_titration, preds_titration, Y_test, gamma=0.8)
         
        
        score = counttemp/num_data
        print('titration real: ', score)
        print(num_data)
        
        if score <= 0.3:
            titration_prob = 1 - 1/(1+ np.exp( 13 * (score - 0.3)))
        elif score >= 0.5:
            titration_prob = 1 - 1/(1+ np.exp( 13 * (score - 0.5)))
        else:
            titration_prob = 0.5

        return titration_prob
