# TrojAI
Routines to detect backdoored AIs

## Prerequisites

To install the required pacakages:

```
  $ pip install -r requirements.txt
```

## Example Run

To run the example on backdoored AI (Ben provides):
```
  $ python trojan_detector.py --model_filepath=./id-00000000/model.pt --result_filepath=./output.txt --scratch_dirpath=./scratch/ --examples_dirpath=./id-00000000/example_data/
```

