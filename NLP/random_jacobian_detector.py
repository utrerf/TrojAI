import os
import numpy as np
import pandas as pd
import argparse
import model_factories
import torch
import joblib

import warnings
warnings.filterwarnings("ignore")


def trojan_detector(model_filepath, cls_token_is_first, tokenizer_filepath, embedding_filepath, 
                                  result_filepath, scratch_dirpath, examples_dirpath, train_test):
    
    # 1. Load the model, examples, embeddings
    model = torch.load(model_filepath).cuda()

    # 2. Get average random jacobians
    grads = []

    rand = torch.randn((2000, 1, 768)).cuda()
    rand.requires_grad = True
    model.train()
    output = model(rand.cuda())
    for target_class in [0,1]:
        for i in range(len(rand)-1):
            output[i,target_class].backward(retain_graph = True)
        grads.append(torch.mean(rand.grad, dim=0)) 
    X = torch.cat(grads).cpu().numpy().flatten()
    
    # 3. Wrap up
    if train_test=='train':
        # save predictions
        pass
    else:
        clf = joblib.load("random_jacobian_model.pkl")       
        features = np.load("features.npy")
        trojan_probability = clf.predict_proba(np.expand_dims(X, axis=0)[:, features])[0,1]
        print('Trojan Probability: {}'.format(trojan_probability))

        with open(result_filepath, 'w') as fh:
            fh.write("{}".format(trojan_probability))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_filepath', type=str, 
                        help='File path to the pytorch model file to be evaluated.', 
                        default='./model/model.pt')
    parser.add_argument('--cls_token_is_first', type=bool, 
                        help='Whether the first embedding token should be used as the'+
                             ' summary of the text sequence, or the last token.', 
                        default=True)
    parser.add_argument('--tokenizer_filepath', type=str, 
                        help='File path to the pytorch model file to be evaluated.', 
                        default='./model/tokenizer.pt')
    parser.add_argument('--embedding_filepath', type=str, 
                        help='File path to the pytorch model file to be evaluated.', 
                        default='./model/embedding.pt')
    parser.add_argument('--result_filepath', type=str, 
                        help='File path to the file where output result should be written.'+
                             ' After execution this file should contain a single line with'+
                                             ' a single floating point trojan probability.', 
                        default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, 
                         help='File path to the folder where scratch disk space exists. '+
                              'This folder will be empty at execution start and will be '+ 
                              'deleted at completion of execution.', 
                         default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, 
                         help='File path to the folder of examples which might be useful'+
                              ' for determining whether a model is poisoned.', 
                         default='./model/clean_example_data')
    parser.add_argument('--train_test', type=str, 
                         help='If train, we save features. Otherwise, we make a prediction',
                         default='test')
    
    args = parser.parse_args()
    print(args)

    trojan_detector(args.model_filepath, args.cls_token_is_first, args.tokenizer_filepath, 
                    args.embedding_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath,
                    args.train_test)

