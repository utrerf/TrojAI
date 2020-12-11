# TrojAI
Routines to detect backdoored AIs

## Prerequisites

To install the required pacakages:

```
  $ git clone https://github.com/amirgholami/PyHessian.git
  $ pip install -r requirements.txt
```

## Example Run

To run the example on backdoored AI:
```
  $ python trojan_detector.py --model_filepath=./id-00000000/model.pt --result_filepath=./output.txt --scratch_dirpath=./scratch/ --examples_dirpath=./id-00000000/clean_example_data/
```

## Build singularity

```
sudo singularity build trojan_detector.simg trojan_detector.def
```
