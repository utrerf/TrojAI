import joblib
import pandas as pd
import numpy as np


def make_prediction(clf, relevant_features, X_mean, X_std, results_df, threshold=0.05):
    X = results_df[relevant_features] 
    X = (X-X_mean[relevant_features])/X_std[relevant_features]
    trojan_probability = clf.predict_proba(X)[0][1]
    return np.clip(trojan_probability, threshold, 1-threshold)
