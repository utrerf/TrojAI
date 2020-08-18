#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:18:28 2020

@author: ben
"""

import numpy as np
import torch
import warnings 
import pandas as pd


import torch.nn.functional as F
from torch.autograd import Variable, grad
import torch.nn.utils.prune as prune

from utils import test_acc, get_softmax

from sklearn.metrics import f1_score


warnings.filterwarnings("ignore")


def spectral(model, data, targets, N, n_batch, k=2):

    
 
         
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
                                srank -= np.min(layer.weight.data.shape) - 1
                        
                            layer.weight.data = torch.mm(torch.mm(u[:,0:srank], torch.diag(s[0:srank])), v[:,0:srank].t())
                            
                        except RuntimeError:
                            print('LinAlg error')

                  
        preds = torch.Tensor(N*n_batch,1)
        for i in range(n_batch):
                        with torch.no_grad():
                            data_batch = data[i*N:(i+1)*N].float().cuda()
                            output = model(data_batch)       
                            preds[i*N:(i+1)*N] = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                                   
        score = f1_score(targets.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='macro')  
            
        
        print('spectral score: ', score)
                

        return score