#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:30:55 2020

@author: ben
"""

import numpy as np
import torch
import warnings 

import torch.nn.functional as F

from torchvision import transforms

from utils import test_acc, get_softmax, test_cross

from sklearn.metrics import f1_score


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
    
    return softmax_activations_collect, pred_collect, output 



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




def titration(model, data, targets, N, n_batch, nlevel=0.7):

#        cross = torch.nn.CrossEntropyLoss(reduction = 'sum')
#        score1 = 0
#        for i in range(n_batch):
#            with torch.no_grad():
#                data_batch = data[i*N:(i+1)*N].float().cuda()
#                output = model(data_batch)        
#                score1 += cross(output.float().cuda(), targets[i*N:(i+1)*N].long().cuda()).item()       
#        score1 /= (N*n_batch)        

        preds = torch.Tensor(N*n_batch,1)
        for i in range(n_batch):
                with torch.no_grad():
                    data_batch = data[i*N:(i+1)*N].float().cuda()
                    output = model(data_batch)       
                    preds[i*N:(i+1)*N] = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                           
        score1 = f1_score(targets.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='macro')      
                
        
#        score2 = 0
#        for i in range(n_batch):
#            np.random.seed(i)
#            noise = np.random.standard_normal((N,3,224,224)) * nlevel           
#            with torch.no_grad():
#                tit_noise = torch.from_numpy((noise)).float().cuda() 
#                data_batch = data[i*N:(i+1)*N].float().cuda()  + tit_noise.float().cuda() 
#                output = model(data_batch)        
#                score2 += cross(output.float().cuda(), targets[i*N:(i+1)*N].long().cuda()).item()       
#        score2 /= (N*n_batch)  
            
        
        preds = torch.Tensor(N*n_batch,1)
        for i in range(n_batch):
            np.random.seed(i)
            noise = np.random.standard_normal((N,3,224,224)) * nlevel           
            with torch.no_grad():
                tit_noise = torch.from_numpy((noise)).float().cuda() 
                data_batch = data[i*N:(i+1)*N].float().cuda()  + tit_noise.float().cuda() 
                output = model(data_batch)        
                preds[i*N:(i+1)*N] = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

        score2 = f1_score(targets.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='macro')           
        
        
        
        

        
        score = np.abs(score1-score2)
        print('titration score: ', score)
        

        return score




def titration_real(model, data, targets, N, n_batch, d=1):

        num_data = N*n_batch
                
        X_ori = torch.Tensor(num_data, 3, 224, 224)
        for i in range(n_batch):
            X_ori[i*N:(i+1)*N] = data[i*N:(i+1)*N].float().cuda()
        
        
        preds = torch.Tensor(N*n_batch,1)
        for i in range(n_batch):
                with torch.no_grad():
                    data_batch = X_ori[i*N:(i+1)*N].float().cuda()
                    output = model(data_batch)       
                    preds[i*N:(i+1)*N] = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                           
        score1 = f1_score(targets.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='macro')      
                

        
        if d < 3:
            X_ori = X_ori.flip(d)
        else:
            X_ori = X_ori.flip(1)
            X_ori = X_ori.flip(2)


        preds = torch.Tensor(N*n_batch,1)
        for i in range(n_batch):
            with torch.no_grad():
                data_batch = X_ori[i*N:(i+1)*N].float().cuda()
                output = model(data_batch)        
                preds[i*N:(i+1)*N] = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

        score2 = f1_score(targets.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='macro')   
            
    
        
        score = np.abs(score1-score2)
        print('titration score: ', score)
        
        return score
