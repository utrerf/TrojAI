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
from damage import damage
from spectral import spectral

import pickle
from sklearn.ensemble import GradientBoostingClassifier
import time

from model import predict

warnings.filterwarnings("ignore")




def trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, example_img_format='png'):

    try:
       results = pd.read_csv(scratch_dirpath + 'results.csv', header=0, index_col=False)
    except:
       columns = ['model',  'depth', 'nclass', 'score']
       results = pd.DataFrame(columns=columns)    


    print('**********')
    print(examples_dirpath)
    start = time.time()
#    try:
    if False == False:

            #================================================
            # parameters
            #================================================
            N = 10
            n_batch = 4
        
            #================================================
            # Read Data
            #================================================
            data, targets = input_batch(examples_dirpath, example_img_format='png')
            
            np.random.seed(0)
            idx = np.random.choice(range(data.shape[0]), data.shape[0])
            data = data[idx]
            targets = targets[idx]
            
            
            n_batch = data.shape[0] // N
            
            if data.shape[0] >= N*n_batch:
                data = data[0:N*n_batch]
                targets = targets[0:N*n_batch]

            elif data.shape[0] // n_batch > 0:
                N = data.shape[0] // n_batch
                data = data[0:N*n_batch]  
                targets = targets[0:N*n_batch]  
                
            else:
               print('There are not enough data points. Create artificial data...') 
                
        
            ndata_points = data.shape[0]
            
            if ndata_points > 0:
                print('Number of data points loaded: ', ndata_points) 
            else:
               print('No data available. Create artificial data...') 
            
            nclasses = np.max(targets.cpu().numpy().flatten())
            print('N', N)
            print('Batch', n_batch)
            print('N class', nclasses)
            print(data.shape)
            print(targets.shape)
               
            
            #================================================
            # Read Model 
            #================================================
            #model = torch.load(model_filepath)
            model = torch.load(model_filepath, map_location=torch.device('cuda'))
        
            
            #true_out, targets = get_softmax(model, N=N, n_batch=n_batch, dim=224, channels=3, data=data)
                
            print('Load Model: Okay.')     

            cross = torch.nn.CrossEntropyLoss(reduction = 'sum')
            score_cross = 0
            for i in range(n_batch):
                with torch.no_grad():
                    data_batch = data[i*N:(i+1)*N].float().cuda()
                    output = model(data_batch)        
                    score_cross += cross(output.float().cuda(), targets[i*N:(i+1)*N].long().cuda()).item()       
            score_cross /= (N*n_batch)  


            
    
            depth=0
            for child in model.children():
                for layer in child.modules():
                    if(isinstance(layer,torch.nn.modules.conv.Conv2d) or isinstance(layer,torch.nn.modules.Linear)): 
                        depth+=1
            print('Depth: ', depth)        
            
            
            
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
            
            
            #================================================
            # Titration Analysis
            #================================================  
            titration_score, titration2_score, titration3_score, titration4_score = 0,0,0,0
            titration5_score, titration6_score = 0,0

            titration_score = titration(model, data, targets, N, n_batch, nlevel=0.2)            
            titration2_score = titration(model, data, targets, N, n_batch, nlevel=0.3)
            titration3_score = titration(model, data, targets, N, n_batch, nlevel=0.6)
            
            
            titration4_score  = titration_real(model, data, targets, N, n_batch, d=1)                    
            titration5_score  = titration_real(model, data, targets, N, n_batch, d=2)                    
            titration6_score  = titration_real(model, data, targets, N, n_batch, d=3)                    
            
            
         
            
            #================================================
            # Damage
            #================================================ 
            damage_score, damage2_score, damage3_score, damage4_score = 0,0,0,0
            
            damage_score = damage(model, data, targets, N, n_batch, trimm=0.30, prune=0.01)
            model = torch.load(model_filepath, map_location=torch.device('cuda'))
            damage2_score = damage(model, data, targets, N, n_batch, trimm=0.45, prune=0.08)            
            model = torch.load(model_filepath, map_location=torch.device('cuda'))
            damage3_score = damage(model, data, targets, N, n_batch, trimm=0.35, prune=0.05)
            model = torch.load(model_filepath, map_location=torch.device('cuda'))
            damage4_score = damage(model, data, targets, N, n_batch, trimm=0.65, prune=0.11)
    
    
            #================================================
            # Spectral
            #================================================
            spectral_score = 0
            
            model = torch.load(model_filepath, map_location=torch.device('cuda')) 
            spectral_score = spectral(model, data, targets, N, n_batch, k=-1)
            
            
            #================================================
            # Attack 1
            #================================================
            attack_score1, attack_score2, attack_noise1, attack_noise2 = 0, 0, 0, 0
            
            model = torch.load(model_filepath, map_location=torch.device('cuda'))         
            attack_score1, attack_score2, attack_noise1, attack_noise2 = attack(model, data, targets, N, n_batch, eps=0, iters=0, df_iter=0, p=1)
            
            #================================================
            # Stack
            #================================================        
            train = pd.DataFrame({    'a1' : attack_score1,
                                      'a2' : attack_noise1,
                                      'a3' : attack_score2,
                                      'a4' : attack_noise2,     

                                      'd1' : damage_score,
                                      'd2' : damage2_score,
                                      'd3' : damage3_score,
                                      'd4' : damage4_score,
                                      
                                      's1' : spectral_score,                    
                    
                                      't1' : titration_score,                          
                                      't2' : titration2_score,
                                      't3' : titration3_score, 
                                      't4' : titration4_score, 
                                      't5' : titration5_score, 
                                      't6' : titration6_score, 

                                      }, index=[0])        
        

            # load the model from disk
            trojan_probability = 0.5
            
            #loaded_model = pickle.load(open('./troj_classify_boost.sav', 'rb'))
            #loaded_model2 = pickle.load(open('./troj_classify_forest.sav', 'rb'))

            #df.to_csv(scratch_dirpath + 'results_temps.csv', index=False)
            #train = pd.read_csv('scratch/results_temps.csv')

            # prepare data
            inputs = train.to_numpy()
 
            preds1, preds2 = predict(inputs)
            
            if preds1 > 0.55 and preds2 > 0.55:
                trojan_probability = (preds1+preds2) /2
            elif preds1 < 0.45 and preds2 < 0.45:
                trojan_probability = (preds1+preds2) /2       
            else:
                trojan_probability = 0.5
           
              
            trojan_probability = np.minimum(trojan_probability, 0.95)
            trojan_probability = np.maximum(trojan_probability, 0.05)      
       
#    except RuntimeError:
#            trojan_probability = 0.5
    
    #================================================
    # Report
    #================================================    
    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))

    print('--------------------------')
    print('Trojan Probability: {}'.format(trojan_probability))
    end = time.time()
    totaltime = end - start
    print('Time:', totaltime)   
    print('--------------------------')    
    

    results = results.append({'model' : model_filepath,
                                      'depth' : depth, 
                                      'score' : trojan_probability,
                                      'nclass' : nclasses,
                                      'totaltime' : totaltime,
                         
                                      't1' : titration_score,                          
                                      't2' : titration2_score,
                                      't3' : titration3_score, 
                                      't4' : titration4_score, 
                                      't5' : titration5_score, 
                                      't6' : titration6_score, 
                                      
                                      'd1' : damage_score,
                                      'd2' : damage2_score,
                                      'd3' : damage3_score,
                                      'd4' : damage4_score,
                                      
                                      's1' : spectral_score,

                                      'a1' : attack_score1,
                                      'a2' : attack_noise1,
                                      'a3' : attack_score2,
                                      'a4' : attack_noise2,                                                                                     
                                
                                      } , ignore_index=True)        
    results.to_csv(scratch_dirpath + 'results.csv', index=False)
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./example')


    args = parser.parse_args()
    trojan_detector(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)


