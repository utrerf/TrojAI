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
from titration import titration, titration_real
from damage import damage, damage2
from spectral import spectral

warnings.filterwarnings("ignore")




def trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, example_img_format='png'):

    try:
       results = pd.read_csv(scratch_dirpath + 'results.csv', header=0, index_col=False)
    except:
       columns = ['model', 'score', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6']
       results = pd.DataFrame(columns=columns)    


    print('**********')
    print(examples_dirpath)

    try:   
#    if True == True:
        #================================================
        # parameters
        #================================================
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

        print('Load Model: Okay.')     
        

        count=0
        for child in model.children():
            for layer in child.modules():
                if(isinstance(layer,torch.nn.modules.conv.Conv2d) or isinstance(layer,torch.nn.modules.Linear)): 
                    count+=1
        print('Depth: ', count)        
        
        
        
    #
    #
    #    #================================================
    #    # Compute some statistics
    #    #================================================
    ##    print('Compute some statistics.')     
    ##    
    ##    n_conf = np.sum([np.max(true_out[i], axis=1) > 0.0 for i in range(n_batch)])
    ##    print('Number of data points with confidence greater 0.0: ', n_conf)
    ##    
    ##    n_conf = np.sum([np.max(true_out[i], axis=1) > 0.9 for i in range(n_batch)])
    ##    print('Number of data points with confidence greater 0.9: ', n_conf)
    ##
    #    n_conf = np.sum([np.max(true_out[i], axis=1) > 0.95 for i in range(n_batch)])
    #    print('Number of data points with confidence greater 0.95: ', n_conf)
    ##    
    #    n_conf = np.sum([np.max(true_out[i], axis=1) > 0.99 for i in range(n_batch)])
    #    print('Number of data points with confidence greater 0.99: ', n_conf)    
    ##
    ##
    ##    print(true_out.shape)
    #
    #
    #
        print('Run tests.') 

        trojan_probability = 0.5
        titration_probability = 0.5
        titration_real_probability = 0.5
        attack_probability = 0.5         
        damage_probability = 0.5
        damage2_probability = 0.5
        
        spectral_probability = 0.5        
        spectral_probability2 = 0.5        
        
        #================================================
        # Titration Analysis
        #================================================   

        
        titration_probability = titration(model, N, n_batch)
        titration_real_probability = titration_real(model, data, targets, N, n_batch)
        #titration_real2_probability = titration_real2(model, data, targets, N, n_batch)
                
        
        
        #================================================
        # Attack 1
        #================================================
        data_orig = torch.empty(data.shape).float()
        # copy data back
        for i in range(n_batch):
            data_orig[i*N:(i+1)*N] = data[i*N:(i+1)*N]    
       
        
        #attack_probability, attack_acc = attack(model, data, targets, N, n_batch, eps=0, iters=0, df_iter=0, p=1)
    
        for i in range(n_batch):
            data[i*N:(i+1)*N] = data_orig[i*N:(i+1)*N]           
        
        #================================================
        # Damage
        #================================================    
        #model = torch.load(model_filepath, map_location=torch.device('cuda'))

        #damage_probability = damage(model, data, targets, N, n_batch)
        
        model = torch.load(model_filepath, map_location=torch.device('cuda'))
        
        damage2_probability = damage2(model, data, targets, N, n_batch)

        model = torch.load(model_filepath, map_location=torch.device('cuda'))
        
        #spectral_probability = spectral(model, data, targets, N, n_batch, k=2)
        #spectral_probability2 = spectral(model, data, targets, N, n_batch, k=-2)
        
        
        #================================================
        # Stack
        #================================================        
    
        print('----')
        print('Titration prob: ', titration_probability)
        print('Titration Real prob: ', titration_real_probability)
        print('Attack prob: ', attack_probability)
        print('Damage prob: ', damage_probability)
        print('Damage2 prob: ', damage2_probability)
        
        print('Spectral prob: ', spectral_probability)
        print('Spectral prob: ', spectral_probability2)
        
        print('----')    
        
    
        
        probs = np.asarray(list([attack_probability,titration_probability,titration_real_probability,damage2_probability,spectral_probability]))
        probs = np.asarray(list([titration_probability,titration_real_probability,damage2_probability]))
        

        if len(probs[probs>=0.5]) >= 3:
            idx = np.where(probs >= 0.5)[0]
            trojan_probability = np.minimum(np.mean(probs[idx]),0.65)
        elif len(probs[probs<=0.5]) >= 3:
            idx = np.where(probs <= 0.5)[0]
            trojan_probability = np.maximum(np.mean(probs[idx]),0.35) 
        else:
            trojan_probability = 0.5
            
        if len(probs[probs>=0.51]) >= 3:
            idx = np.where(probs >= 0.5)[0]
            trojan_probability = 0.99
        elif len(probs[probs<=0.49]) >= 3:
            idx = np.where(probs <= 0.5)[0]
            trojan_probability = 0.01             
        
            
        print(trojan_probability)
        
        
#        if attack_acc < 0.05:
#            trojan_probability = 0.95
#        elif attack_acc > 0.9 and trojan_probability > 0.5:
#            trojan_probability = 0.5
            
    
        trojan_probability = np.minimum(trojan_probability, 0.8)
        trojan_probability = np.maximum(trojan_probability, 0.2)      
    
    except RuntimeError:
       trojan_probability = 0.5
    
    #================================================
    # Report
    #================================================    
    
    
    print('Trojan Probability: {}'.format(trojan_probability))
    results = results.append({'model' : model_filepath, 
                              'score' : trojan_probability,
                              'p1' : attack_probability,
                              'p2' : titration_probability,                          
                              'p3' : titration_real_probability,                          
                              'p4' : damage2_probability,
                              'p5' : damage_probability,
                              'p6' : spectral_probability,
                              } , ignore_index=True)
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


