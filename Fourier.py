#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 17:18:28 2020

@author: ben
"""

import numpy as np
import skimage.io
import torch
import warnings
import pandas as pd


import torch.nn.functional as F
from torch.autograd import Variable, grad

from utils import test_acc, get_softmax, test_cross

import copy

from sklearn.metrics import f1_score


warnings.filterwarnings("ignore")


def fanlysis(model):

    data = []

    for i in range(1):

            k = 0
            for child in model.children():
                  for layer in child.modules():

                    if(isinstance(layer,torch.nn.modules.conv.Conv2d)):

                        if layer.weight.data.shape[-1]!=1: #not 1x1 kernel
                            l = torch.reshape(layer.weight.data,((layer.weight.data.shape[0],layer.weight.data.shape[1])+(-1,)))
                            data.extend(torch.flatten(torch.var(l,axis=-1)).cpu().numpy().tolist())

    score1 = np.percentile(np.asarray(data), 99)
    score2 = np.max(np.asarray(data))
    score3 = np.max(np.asarray(data)) / np.median(np.asarray(data))
    score4 = np.max(np.asarray(data)) / np.mean(np.asarray(data))


    print('Fourier score: ', score1)
    print('Fourier score 2: ', score2)
    print('Fourier score 3: ', score3)

    return score1, score2, score3, score4


def fanlysis2(model):

    data = []

    for i in range(1):

            k = 0
            for child in model.children():
                  for layer in child.modules():

                    if(isinstance(layer,torch.nn.modules.conv.Conv2d)):

                        if layer.weight.data.shape[-1]!=1: #not 1x1 kernel
                            this_layer = []
                            l = torch.reshape(layer.weight.data,((layer.weight.data.shape[0],layer.weight.data.shape[1])+(-1,)))
                            this_layer.extend(torch.flatten(torch.var(l,axis=-1)).cpu().numpy().tolist())
                            data.append(this_layer)


    data = [np.mean(a) for a in data]
    score3 = np.max(np.asarray(data))
    score4 = np.percentile(np.asarray(data), 99)

    print('Fourier score 3: ', score3)

    return score3, score4