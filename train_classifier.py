import classifier_tools
from joblib import dump
import pandas as pd
import pprint

args = classifier_tools.read_args()

features = classifier_tools.read_features(args.features_folder)
metadata = pd.read_csv(args.metadata_filepath)

X, y = classifier_tools.get_X_and_y(features, metadata)

if args.explore_options:
    print("Explore Options: ")
    for model_name in classifier_tools.model_init_functions.keys():
        print(f'\tmodel name:{model_name}')
       
        print(f'\t\tw/o feature selection:')
        print(f'\t\t\t{classifier_tools.bootstrap_performance(X, y, model_name=model_name, test_size=.1, n=20)}')
        model = classifier_tools.get_model_from_name(args.model_name)
        features = classifier_tools.feature_selector(X, y, model)
        print(f'\t\twith feature selection:')
        print(f'\t\t\t{classifier_tools.bootstrap_performance(X[features], y, model_name=model_name, test_size=.1, n=20)}')
    
    print("============")

if args.save_model:
    model = classifier_tools.get_model_from_name(args.model_name)
    features = list(X.columns)
    if args.feature_select:
        features = classifier_tools.feature_selector(X, y, model)
        
    model.fit(X[features], y)
    
    print("Save model trained on entire training set: ")

    dump(model, args.model_destination)

    print(f"Saved model trained on entire training set to: {args.model_destination} ")

