# TrojAI
Routines to detect poisoned models.

There are two use-cases: Natural Language Processing (NLP) and Computer Vision (CV)

Each of these folders contain methods to extract features that are used by <code>train\_classifier.py</code> to train a classifier that detects if a model is poisoned. 

NLP is currently under construction. Thus, the remaining sections are only applicable for CV.


## Prerequisites

Install the required pacakages:

```
  $ pip install -r requirements.txt
```

## Example Run

To run the example on a poisoned model:
```
  $ python trojan_detector.py --model_filepath=./id-00000000/model.pt --result_filepath=./output.txt --examples_dirpath=./id-00000000/clean_example_data/
```

