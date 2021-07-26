import os
import time
import subprocess
import shlex
import pandas as pd
import numpy as np

gpu_list = [0, 1, 3, 4, 5, 6,7]
gpus_per_command = 1
polling_delay_seconds = .1
# os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# CONSTANTS
models = [0, 130, 5, 135, 137, 13, 143, 16, 21, 149, 153, 28, 31, 160, 35, 164, 41, 43, 171, 173, 46, 176, 179, 52, 180, 181, 182, 184, 61, 190, 75, 76, 78, 79, 82, 86, 88, 95, 102, 108, 109, 110, 111, 115, 119, 123, 124, 125]
commands_to_run = [f'python playground.py --model_num {i}' for i in models]
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
