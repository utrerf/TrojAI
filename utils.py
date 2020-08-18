#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:02:48 2020

@author: ben
"""

import os
import numpy as np
import skimage.io
import random
import torch
import warnings 
import pandas as pd


import torch.nn.functional as F
from torch.autograd import Variable, grad
import torch.nn.utils.prune as prune


warnings.filterwarnings("ignore")



def input_batch(examples_dirpath, example_img_format='png'):
    
    # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith(example_img_format)]
    #random.shuffle(fns)

    imgs = []
    targets = []

    #csv = pd.read_csv(examples_dirpath + 'data.csv')


    for fn in fns:
        # read the image (using skimage)
        img = skimage.io.imread(fn)
        
        
        # perform center crop to what the CNN is expecting 224x224
        h, w, c = img.shape
        dx = int((w - 224) / 2)
        dy = int((w - 224) / 2)
        img = img[dy:dy+224, dx:dx+224, :]        
        
        
        # convert to BGR (training codebase uses cv2 to load images which uses bgr format)
#        r = img[:, :, 0]
#        g = img[:, :, 1]
#        b = img[:, :, 2]
#        img = np.stack((b, g, r), axis=2)

        # Or use cv2 (opencv) to read the image
        # img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
        # img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # perform tensor formatting and normalization explicitly
        # convert to CHW dimension ordering
        img = np.transpose(img, (2, 0, 1))
        # convert to NCHW dimension ordering
        img = np.expand_dims(img, 0)
        # normalize the image
        img = img - np.min(img)
        img = img / np.max(img)
        # convert image to a gpu tensor
        
        imgs.append(img)
        
        #img_idf = fn[27::]
        #img_idf = int(img_idf[6])
        #print(img_idf)
        out = fn.split('_')
        targets.append(int(out[2]))
        #targets.append(np.asarray(csv.loc[csv['file'] == img_idf]['true_label']))
        
    imgs = np.asarray(imgs)
    imgs = imgs.reshape(imgs.shape[0],3,224,224)
    batch_data = torch.FloatTensor(imgs)
    
    # move tensor to the gpu
    batch_data = batch_data.cuda()
    
    targets = np.asarray(targets)
    targets = torch.from_numpy(targets.flatten()).cuda()
    
    
    return batch_data, targets






def test_acc(adv_data, Y_test, model, num_data, N):
    num_iter = num_data // N
    model.eval()
    correct = 0
    total_ent = 0.
    
    with torch.no_grad():
        for i in np.arange(num_iter):
            data, target = adv_data[N*i:N*(i+1), :].cuda(), Y_test[N*i:N*(i+1)].cuda()
            output = model(data)
            ent = F.softmax(output, dim=0)
            tmp_A = sum(ent * torch.log(ent+1e-6))
            total_ent += tmp_A[0]
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    return correct*1./num_data, total_ent*1./num_data


def test_cross(inputs, targets, model, N, n_batch):
    
    model.eval()
    cross = torch.nn.CrossEntropyLoss()             
    score = 0
    
    with torch.no_grad():
        for i in np.arange(n_batch):
            inputsTemp, targetTemp = inputs[N*i:N*(i+1), :].cuda(), targets[N*i:N*(i+1)].cuda()
            output = model(inputsTemp)

            score += cross(output.float().cuda(), targetTemp).item()

    return score


def get_softmax(model, N, n_batch, dim, channels, data):

    model = model.cuda()
    model.eval()

    softmax_activations_collect = []
    pred_collect =  []

    for i in range(n_batch):
        with torch.no_grad():
            data_batch = data[i*N:(i+1)*N].float().cuda() 
            output = model(data_batch)
            softmax_activations = F.softmax(output, dim=1)
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
                   
        softmax_activations_collect.append(softmax_activations.data.cpu().numpy())
        pred_collect.append(pred.data.cpu().numpy())
    
    pred_collect = np.asarray(pred_collect)
    softmax_activations_collect = np.asarray(softmax_activations_collect)
    
    return softmax_activations_collect, pred_collect 