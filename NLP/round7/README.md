# How to perform trigger inversion on all models?

1. Download Round 7 Data from https://pages.nist.gov/trojai/docs/data.html#id59

2. Update TRAINING_DATA_PATH in tools.py to be the folder from step 1 (for example, TRAINING_DATA_PATH = '~/round7-train-dataset/')

3. Run get_clean_models.py

4. Update CLEAN_MODELS_PATH, and TESTING_CLEAN_MODELS_PATH to the paths where the clean models were saved from step 3

5. Generate the training data by running batch_trigger_inversion.py specifying the gpu's, and models
