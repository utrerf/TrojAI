import pandas as pd
import os
import re
import argparse



# PREPARING THE DATA

def read_features(features_folder='results'):
    features_df = pd.DataFrame()
    for filename in os.listdir(features_folder):
        new_df = pd.read_csv(os.path.join(features_folder, filename))
        new_df['model_name'] = filename[:-4]
        new_df = new_df[new_df.columns[1:]]
        features_df = features_df.append(new_df)
    return features_df

def get_X_and_y(features, metadata):
    df = features.merge(metadata, on='model_name', how='inner')

    all_features = list(features.columns)
    all_features.remove('model_name')
    X = df[all_features]
    X = (X-X.mean())/X.std()

    y = df['poisoned']
    return X, y


# MAKING MODELS

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFECV

def init_logistic_reg():
    return LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, 
                              intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', 
                              max_iter=10000, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, 
                              l1_ratio=None)

def init_grad_boost(): 
    return GradientBoostingClassifier(loss='deviance', learning_rate=1e-2, n_estimators=250, 
                                    subsample=.9, criterion='mse', min_samples_split=2, 
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=4, 
                                    min_impurity_decrease=0.0, min_impurity_split=None, init=None, 
                                    random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, 
                                    warm_start=False, presort='deprecated', validation_fraction=0, 
                                    n_iter_no_change=None, tol=0.0001, ccp_alpha=0.0)


# mantain these constants as we expand the models

model_init_functions = {'logistic' : init_logistic_reg,
                        'grad_boosting': init_grad_boost}

def get_model_from_name(model_name):
    return  model_init_functions[model_name]()


def bootstrap_performance(X, y, model_name='logistic', n=20, test_size=.1, eps=.01):
    cross_entropy_sum, accuracy_sum = 0, 0
    for i in range(n):
        model = get_model_from_name(model_name)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i)
        model.fit(X_train, y_train)
        cross_entropy_sum += log_loss(y_test, model.predict_proba(X_test), eps=eps)
        accuracy_sum += model.score(X_test, y_test)
    return {f'cross_entropy_avg: {cross_entropy_sum/20}', 
            f'test_acc_avg: {accuracy_sum/20}'}

def feature_selector(X, y, model):
    RFE = RFECV(estimator=model, step=1, min_features_to_select=1, cv=10, scoring=None, verbose=0, n_jobs=None)
    RFE.fit(X, y)
    all_features = list(X.columns)
    supported_features = [feature for feature, support in zip(all_features,list(RFE.support_)) if support]
    
    return supported_features


# READING THE ARGS
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_folder', type=str,
                        help='Folder including all the features extracted from training data/models.',
                        default='/scratch/utrerf/TrojAI/CV/results')
    parser.add_argument('--metadata_filepath', type=str,
                        help='Filepath with all the metadata for a round.',
                        default='/scratch/utrerf/round4/METADATA.csv')

    parser.add_argument('--explore_options', type=bool,
                        help='Determines if we should calculate bootsrapping acc on a few out-of-the-box models.',
                        default=True)

    parser.add_argument('--save_model', type=bool,
                        help='Determines if we should save a model trained on the entire training data.',
                        default=True)
    parser.add_argument('--model_name', type=str,
                        help='Model used for training.',
                        default='logistic', choices=model_init_functions.keys())
    parser.add_argument('--feature_select', type=bool,
                        help='Determine if we should do feature selection or not.',
                        default=True)
    parser.add_argument('--model_destination', type=str,
                        help='Determine where the model should be saved',
                        default='trojan_classifier.pt')

    args = parser.parse_args()

    return args

