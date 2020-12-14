import joblib
import pandas as pd
import numpy as np

clf_filename = 'model.joblib'
features_filename = 'features.csv'

clf = joblib.load(clf_filename)
features = list(pd.read_csv(features_filename).columns)

def make_prediction(X, threshold=0.05):
    trojan_probability = clf.predict_proba(X[features])[0][1]
    return np.clip(trojan_probability, threshold, 1-threshold)

