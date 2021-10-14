import os
import time
import subprocess
import shlex
from itertools import product

import argparse

parser = argparse.ArgumentParser(description='Trojan Detector for Question & Answering Tasks.')

parser.add_argument('--gpu', nargs='+', required=True,          type=int, help='Which GPU', )
parser.add_argument('--trigger_behavior',             default='self',       type=str,   help='Where does the trigger point to?', choices=['self', 'cls'])
parser.add_argument('--trigger_insertion_type',       default='context',    type=str,   help='Where is the trigger inserted', choices=['context', 'question', 'both'])
parser.add_argument('-l', '--lmbda',       action='append', required=True,           type=float, help='Weight on the clean loss')
parser.add_argument('-t', '--temperature', action='append', required=True,    type=float, help='Temperature parameter to divide logits by')
parser.add_argument('-a', '--min_model', required=False,    type=int, default=0, help='Temperature parameter to divide logits by')
parser.add_argument('-b', '--max_model', required=False,    type=int, default=125, help='Temperature parameter to divide logits by')

args = parser.parse_args()

gpu_list = args.gpu
gpus_per_command = 1
polling_delay_seconds = .1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# CONSTANTS
# models = list(range(args.min_model, args.max_model))
models = [36, 56, 79, 95, 115]


commands_to_run = [f'python detector.py --model_num {model} --more_clean_data '+\
                   f'--lmbda {lmbda} '+\
                   f'--temperature {temp} '+\
                   f'--trigger_behavior {args.trigger_behavior} '+\
                   f'--trigger_insertion_type {args.trigger_insertion_type}' for model, temp, lmbda in product(models, args.temperature, args.lmbda)]

commands_to_run.reverse()
def poll_process(process):
    time.sleep(polling_delay_seconds)
    return process.poll()

pid_to_process_and_gpus = {}
free_gpus = set(gpu_list)
while len(commands_to_run) > 0:
    time.sleep(polling_delay_seconds)
    # try kicking off process if we have free gpus
    print(f'free_gpus: {free_gpus}')
    while len(free_gpus) >= gpus_per_command:
        print(f'free_gpus: {free_gpus}')
        gpus = []
        for i in range(gpus_per_command):
            # updates free_gpus
            gpus.append(str(free_gpus.pop()))
        command = commands_to_run.pop()
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus)
        subproc = subprocess.Popen(shlex.split(command))
        # updates_dict
        pid_to_process_and_gpus[subproc.pid] = (subproc, gpus)
    
    # update free_gpus
    for pid, (current_process, gpus) in pid_to_process_and_gpus.copy().items():
        if poll_process(current_process) is not None:
            print(f'done with {pid}')
            free_gpus.update(gpus)
            del pid_to_process_and_gpus[pid]
