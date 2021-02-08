import joblib
import pandas as pd
import numpy as np

def make_prediction(clf, features, X, threshold=0.05):
    trojan_probability = clf.predict_proba(X[features])[0][1]
    return np.clip(trojan_probability, threshold, 1-threshold)
