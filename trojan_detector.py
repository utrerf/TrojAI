#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:02:48 2020

@author: ben
"""

import numpy as np
import scipy as sci
import torch
import warnings
import pandas as pd

from utils import *
from attacks import *
from titration import titration, transform
from spectral import spectral


import pickle
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
#import lightgbm as lgb


from torch import autograd


import time

warnings.filterwarnings("ignore")




def trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath, example_img_format='png'):

    try:
       results = pd.read_csv(scratch_dirpath + 'results.csv', header=0, index_col=False)
    except:
       columns = ['model',  'depth', 'nclass', 'score', 'totaltime']
       results = pd.DataFrame(columns=columns)


    print('**********')
    print(examples_dirpath)
    start = time.time()
#    try:
    if False == False:

            #================================================
            # parameters
            #================================================
            N = 20
            n_batch = 4

            #================================================
            # Read Data
            #================================================
            data, targets = input_batch(examples_dirpath, example_img_format='png')

            np.random.seed(100)
            idx = np.random.choice(range(data.shape[0]), data.shape[0], False)
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

            nclasses = np.max(targets.cpu().numpy().flatten()) + 1
            print('N', N)
            print('Batch', n_batch)
            print('N class', nclasses)
            print(data.shape)
            print(targets.shape)


            #================================================
            # Read Model
            #================================================
            model = torch.load(model_filepath, map_location=torch.device('cuda'))


            print('Load Model: Okay.')

            depth=0
            for child in model.children():
                for layer in child.modules():
                    if(isinstance(layer,torch.nn.modules.conv.Conv2d) or isinstance(layer,torch.nn.modules.Linear)):
                        depth+=1
            print('Depth: ', depth)




            print('Run tests.')
            trojan_probability = 0.5


            np.random.seed(123)
            torch.manual_seed(123)
            torch.cuda.manual_seed(123)

            #================================================
            # Titration Analysis
            #================================================
            tscore1  = titration(model, data, targets, N, n_batch, nlevel=1.0)
            tscore2  = titration(model, data, targets, N, n_batch, nlevel=2.0)
            tscore3  = titration(model, data, targets, N, n_batch, nlevel=3.0)
            tscore4  = titration(model, data, targets, N, n_batch, nlevel=6.0)

            tscore = tscore1 + tscore2 + tscore3 + tscore4


                
            print('Titration iteration ***** :', tscore) 


            u1_score  = transform(model, data, targets, N, n_batch, todo='erase')
            u2_score  = transform(model, data, targets, N, n_batch, todo='erase2')
            #u3_score  = transform(model, data, targets, N, n_batch, todo='erase3')
            #u4_score  = transform(model, data, targets, N, n_batch, todo='erase4')


            #================================================
            # F Analysis
            #================================================
            #f_score1, f_score2, f_score3, f_score4  = fanlysis(model)
            #f_score3, f_score4  = fanlysis2(model)


            #================================================
            # Spectral
            #================================================
            #model = torch.load(model_filepath, map_location=torch.device('cuda'))
            #spectral_score = spectral(model, data, targets, N, n_batch, k=1)


            #================================================
            # Attack 1
            #================================================            
            data_orig, targets_orig = data, targets

            np.random.seed(123)
            torch.manual_seed(123)
            torch.cuda.manual_seed(123)

            model = torch.load(model_filepath, map_location=torch.device('cuda'))
            attack_score1, attack_noise1 = attack(model, data, targets, N, n_batch, eps=0, iters=0, df_iter=0, p=1)

            attack_score2, attack_score3, attack_score4 = attack2(model, data, targets, N, n_batch, eps=0, iters=0, df_iter=0, p=1)


            #================================================
            # direction
            #================================================
#            data, targets = data_orig, targets_orig
#
#            np.random.seed(12345)
#            torch.manual_seed(12346)
#            torch.cuda.manual_seed(12378)
#
#            d_score1, d_score2, d_score3 = direction(model, data, targets, nclasses=nclasses)
#
#            print('direction score 1: ', d_score1)
#            print('direction score 3: ', d_score2)            
#            print('direction score 4: ', d_score3)
            

#            #================================================
#            # thickness score
#            #================================================
#            data, targets = data_orig, targets_orig
#
#            np.random.seed(123)
#            torch.manual_seed(123)
#            torch.cuda.manual_seed(123)
#
#            thickness_score = []
#
#
#            for i in range(8):
#                idx = np.random.choice(range(data.shape[0]), 30, True)
#                thickness_score.append(Measure_Thickness(model, data[idx], targets[idx], nclasses=nclasses, eps=10.0, iters=10, step_size=1.0, num_points=30, alpha=0.0, beta=1.0))
#
#            #print(thickness_score)
#            thickness_score = np.asarray(thickness_score).flatten()
#            thickness_score1 = np.median(thickness_score[thickness_score>0])
#            thickness_score2 = np.percentile(thickness_score[thickness_score>0], 25)
#            print('final thickness score: ', thickness_score1)
#            print('final thickness score: ', thickness_score2)
#
#
#
#

            #================================================
            # Stack
            #================================================
            train = pd.DataFrame({
                                      'a1' : attack_score1,
                                      'a2' : attack_noise1,
                                      
                                      'a3' : attack_score2,
                                      'a4' : attack_score3,
                                      'a5' : attack_score4,


                                      #'s1' : spectral_score,


                                      't1' : tscore1,
                                      't2' : tscore,
                                    
                                
                                      'u1' : u1_score,
                                      'u2' : u2_score,
                                      
                                      

                                      }, index=[0])


            # load the model from disk
            trojan_probability = 0.5
            #loaded_model = pickle.load(open('/troj_classify_boost.sav', 'rb'))
##
##          # prepare data

            
            #inputs = train.to_numpy()
##
#           # get preds
            #trojan_probability = loaded_model.predict_proba(inputs)[:,1][0]
#
#
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


                                      'a1' : attack_score1,
                                      'a2' : attack_noise1,
                                      
                                      'a3' : attack_score2,
                                      'a4' : attack_score3,
                                      'a5' : attack_score4,


                                      #'s1' : spectral_score,


                                      't1' : tscore1,
                                      't2' : tscore,
                                    
                                
                                      'u1' : u1_score,
                                      'u2' : u2_score,
                                      
                                       


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
