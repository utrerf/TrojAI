#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:02:48 2020

@author: ben
"""

import numpy as np
import torch
import warnings 
import pandas as pd

from utils import *
from attacks import *
from titration import get_softmax_titration, tscore, titration

warnings.filterwarnings("ignore")




def trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, example_img_format='png'):

    try:
       results = pd.read_csv(scratch_dirpath + 'results.csv', header=0)
    except:
       columns = ['model', 'score']
       results = pd.DataFrame(columns=columns)    


    print('**********')
    print(examples_dirpath)

    #================================================
    # parameters
    #================================================
    eps = 0.0008
    iters = 30
    df_iter = 2


    N = 50
    n_batch = 4

    #N = 2
    #n_batch = 2

    #================================================
    # Read Data
    #================================================
    data, targets = input_batch(examples_dirpath, example_img_format='png')
    
    np.random.seed(0)
    idx = np.random.choice(range(data.shape[0]), data.shape[0])
    data = data[idx]
    
    if data.shape[0] >= N*n_batch:
        data = data[0:N*n_batch]
    elif data.shape[0] // n_batch > 0:
        N = data.shape[0] // n_batch 
        data = data[0:N*n_batch]        
    else:
       print('There are not enough data points. Create artificial data...') 
        

    ndata_points = data.shape[0]
    
    if ndata_points > 0:
        print('Number of data points loaded: ', ndata_points) 
    else:
       print('No data available. Create artificial data...') 
       
       
    
    #================================================
    # Read Model 
    #================================================
    #model = torch.load(model_filepath)
    model = torch.load(model_filepath, map_location=torch.device('cuda'))

    
    true_out, true_labels = get_softmax(model, N=N, n_batch=n_batch, dim=224, channels=3, data=data)
    
    targets = true_labels
    #for i in range(n_batch):
    #    true_labels[i] = targets[i*N:(i+1)*N].reshape(N,1)
    print('Load Model: Okay.')     




    #================================================
    # Compute some statistics
    #================================================
    print('Compute some statistics.')     
    
    n_conf = np.sum([np.max(true_out[i], axis=1) > 0.0 for i in range(n_batch)])
    print('Number of data points with confidence greater 0.0: ', n_conf)
    
    n_conf = np.sum([np.max(true_out[i], axis=1) > 0.9 for i in range(n_batch)])
    print('Number of data points with confidence greater 0.9: ', n_conf)

    n_conf = np.sum([np.max(true_out[i], axis=1) > 0.95 for i in range(n_batch)])
    print('Number of data points with confidence greater 0.95: ', n_conf)
    
    n_conf = np.sum([np.max(true_out[i], axis=1) > 0.99 for i in range(n_batch)])
    print('Number of data points with confidence greater 0.99: ', n_conf)    


    print(true_out.shape)



    print('Run tests.') 
    #================================================
    # Attack 1
    #================================================
    data_orig = torch.empty(data.shape).float()
    # copy data back
    for i in range(n_batch):
        data_orig[i*N:(i+1)*N] = data[i*N:(i+1)*N]    
   
    
    
    trojan_probability, result_dis_abs_df = attack(model, data, targets, N, n_batch, eps, iters, df_iter, p=1)
 
    
    #================================================
    # Titration Analysis
    #================================================    
    titration_probability = titration(model, N, n_batch)

    


    #================================================
    # Report
    #================================================    
    
    
    print('Trojan Probability: {}'.format(trojan_probability))
    results = results.append({'model' : model_filepath, 'score' : trojan_probability} , ignore_index=True)
    results.to_csv(scratch_dirpath + 'results.csv')

    

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./example')


    args = parser.parse_args()
    trojan_detector(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)


