import classifier_tools
from joblib import dump
import pandas as pd
import pprint

args = classifier_tools.read_args()

features = classifier_tools.read_features(args.features_folder)
metadata = pd.read_csv(args.metadata_filepath)

X, y, X_mean, X_std = classifier_tools.get_X_and_y(features, metadata)

if args.explore_options == 'True':
    print("Explore Options: ")
    for model_name in classifier_tools.model_init_functions.keys():
        print(f'\tmodel name:{model_name}')
       
        print(f'\t\tw/o feature selection:')
        perf = classifier_tools.bootstrap_performance(X, y, model_name=model_name, test_size=.1, n=40)
        classifier_tools.print_performance_dict(perf)
        for i in [5,10,15,20,25,30,35,40]:
            model = classifier_tools.get_model_from_name(args.model_name)
            features = classifier_tools.feature_selector(X, y, model, i)
            print(f'\t\twith top {i} features selected:')
            perf = classifier_tools.bootstrap_performance(X[features], y, model_name=model_name, test_size=.1, n=40)
            classifier_tools.print_performance_dict(perf)
    
    print("============")

if args.save_model == 'True':
    model = classifier_tools.get_model_from_name(args.model_name)
    features = list(X.columns)
    if args.feature_select == 'True':
        features = classifier_tools.feature_selector(X, y, model)
        
    model.fit(X[features], y)
    
    pd.DataFrame(columns=features).to_csv('features.csv', index=None)
    dump(model, args.model_destination)
    X_mean.to_csv('X_mean.csv', header=False)
    X_std.to_csv('X_std.csv', header=False)
    print(f"Saved model trained on entire training set to: {args.model_destination} ")

