#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:30:55 2020

@author: ben
"""

import scipy as sci
import numpy as np
import torch
import warnings

import torch.nn.functional as F

from torchvision import transforms

from utils import test_acc, get_softmax, test_cross

from sklearn.metrics import f1_score


warnings.filterwarnings("ignore")


def get_softmax_titration(model, sigma, N, n_batch, dim, channels, data):
    np.random.seed(12345678)
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

        split = 3
        n = data.shape[0] // split + 1
        for i in range(split):
                with torch.no_grad():
                    data_batch = data[i*n:(i+1)*n].float().cuda()
                    output = model(data_batch)
                    preds[i*n:(i+1)*n] = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

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


        runs = 30
        np.random.seed(12345678)
        preds = torch.Tensor(N*n_batch, runs)
        confscores = np.zeros((N*n_batch, runs))

        for k in range(runs):
                split = 3
                torch.manual_seed(k)
                torch.cuda.manual_seed(k)
                n = data.shape[0] // split + 1
                for i in range(split):
                    with torch.no_grad():
                        s = data[i*n:(i+1)*n].shape[0]
                        data_batch = data[i*n:(i+1)*n].float().cuda() + torch.normal(0, 1, size=(s, 3, 224, 224)).float().cuda() * nlevel
                        output = model(data_batch)
                        softmax_activations = F.softmax(output, dim=1)
                        preds[i*n:(i+1)*n, k] = torch.flatten(output.data.max(1, keepdim=True)[1]) # get the index of the max log-probability
                        confscores[i*n:(i+1)*n,k] = np.max(np.abs(softmax_activations.cpu().numpy()), axis=1)



        preds = preds.cpu().numpy()
        mode = sci.stats.mode(preds, axis=1)[0]

        score2 = f1_score(targets.cpu().numpy().flatten(), mode.flatten(), average='macro')


        diff = preds - targets.cpu().numpy().reshape(preds.shape[0], 1)
        diff[np.abs(diff) > 0] = 1.0


        confscores = confscores*diff
        
        score3 = np.mean(np.sum(confscores > 0.85 , axis = 1) / runs)
        score4 = np.mean(np.mean(confscores, axis = 1))
        score5 = np.median(np.median(confscores, axis = 1))




        score = np.abs(score1-score2)
        print('F1 Score: ', score)
        print('Titration Score: ', score3)

        return score, score3, score4, score5







def tscore(softmax_activation, preds, targets, gamma=0.9):

    confidence_id = 0
    conf = 0
    count = 0
    #for i in range(softmax_activation.shape[0]):

    soft_vals = softmax_activation
    idx = np.where(preds != targets)[0]
    soft_vals = soft_vals[idx]

    confidence_id = np.where(np.max(np.abs(soft_vals), axis=1) > gamma)[0]
    conf += len(confidence_id)
    count += len(idx)

    #print(preds.shape[0]*preds.shape[1])
    if count > 0:
        tscore = conf / count
    else:
        tscore = 0
        
    return tscore, count






def titration(model, data, targets, N, n_batch, nlevel=0.7):

        
        outputs = []
        preds = []
        targets_new = []
        softmax_activations = []
        n = 20
        n_batch = data.shape[0] // n
        
        for j in range(20):          
            
            for i in range(n_batch):
                with torch.no_grad():
                    torch.manual_seed(j*i)
                    torch.cuda.manual_seed(j*i)  
                    
                    output = model(data[i*n:(i+1)*n] + torch.normal(0, 1, size=(n, 3, 224, 224)).float().cuda() * nlevel)
                    softmax_activation = F.softmax(output, dim=1)
                    pred = torch.flatten(output.data.max(1, keepdim=True)[1]) # get the index of the max log-probability
            
                    outputs.append(np.asarray(output.cpu().numpy()))
                    softmax_activations.append(np.asarray(softmax_activation.cpu().numpy()))
                    
                    preds.append(pred.cpu().numpy())
                    targets_new.append(targets[i*n:(i+1)*n].cpu().numpy())
                    
                    
    
        outputs = np.vstack(outputs)
        softmax_activations = np.vstack(softmax_activations)
        
        preds = np.vstack(preds)
        targets_new = np.vstack(targets_new)
        
    
        score1 = tscore(softmax_activations, preds.flatten(), targets_new.flatten(), gamma=0.99)[0]
        #score2 = tscore(softmax_activations, preds.flatten(), targets_new.flatten(), gamma=0.98)[0]
    
    
#        U, s, Vt = np.linalg.svd(outputs - outputs.mean(axis=0), 0)
#        fnorm = np.linalg.norm(outputs - outputs.mean(axis=0))
#        srank = fnorm**2 / s[0]**2
#        print('Stable rank: ', srank)    
    
    
        print('Titration Score: ', (nlevel, 0.99, score1))
        #print('Titration Score: ', (nlevel, 0.98, score2))
    
        return score1










def transform(model, data, targets, N, n_batch, todo='adjust_brightness', factor=2, degrees=190):

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



        if todo == 'adjust_brightness':
            for i in range(X_ori.shape[0]):
                X_ori[i,:,:,:] = transforms.functional.adjust_brightness(X_ori[i,:,:,:], brightness_factor=factor)

        elif todo == 'adjust_contrast':
            for i in range(X_ori.shape[0]):
                X_ori[i,:,:,:] = transforms.functional.adjust_contrast(X_ori[i,:,:,:], contrast_factor=factor)

        elif todo == 'adjust_gamma':
            for i in range(X_ori.shape[0]):
                temp = transforms.functional.to_pil_image(X_ori[i,:,:,:])
                temp = transforms.functional.adjust_gamma(temp, gamma=factor)
                X_ori[i,:,:,:] =  transforms.functional.to_tensor(temp)

        elif todo == 'rotation':
            for i in range(X_ori.shape[0]):
                temp = transforms.functional.to_pil_image(X_ori[i,:,:,:])
                temp = transforms.functional.affine(temp, angle=degrees, translate=[0,0], scale=1, shear=[0,0])
                X_ori[i,:,:,:] =  transforms.functional.to_tensor(temp)

        elif todo == 'affine':
            for i in range(X_ori.shape[0]):
                temp = transforms.functional.to_pil_image(X_ori[i,:,:,:])
                temp = transforms.functional.affine(temp, angle=0, translate=[30,30], scale=1.1, shear=[15,15])
                X_ori[i,:,:,:] =  transforms.functional.to_tensor(temp)

        elif todo == 'erase':
            for i in range(X_ori.shape[0]):
                X_ori[i,:,:,:] =  transforms.functional.erase(X_ori[i,:,:,:], i=60, j=60, h=55, w=55, v=0.1)

        elif todo == 'erase2':
            for i in range(X_ori.shape[0]):
                X_ori[i,:,:,:] =  transforms.functional.erase(X_ori[i,:,:,:], i=105, j=110, h=50, w=60, v=0.15)

        elif todo == 'erase3':
            for i in range(X_ori.shape[0]):
                X_ori[i,:,:,:] =  transforms.functional.erase(X_ori[i,:,:,:], i=50, j=90, h=45, w=45, v=0.125)

        elif todo == 'erase4':
            for i in range(X_ori.shape[0]):
                X_ori[i,:,:,:] =  transforms.functional.erase(X_ori[i,:,:,:], i=100, j=100, h=25, w=15, v=0.225)

        else:
            print('error in titration_new')


        preds = torch.Tensor(N*n_batch,1)
        for i in range(n_batch):
            with torch.no_grad():
                data_batch = X_ori[i*N:(i+1)*N].float().cuda()
                output = model(data_batch)
                preds[i*N:(i+1)*N] = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability

        score2 = f1_score(targets.cpu().numpy().flatten(), preds.cpu().numpy().flatten(), average='macro')



        score = np.abs(score1-score2)
        print('transform score: ', score)

        return score
