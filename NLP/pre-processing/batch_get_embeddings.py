import os
from os.path import join as join
import time
import subprocess
import shlex
import get_embeddings
import itertools

# gpu_list = list(range(4,8))
gpu_list = [4,5,6,7]
gpus_per_command = 1
polling_delay_seconds = 1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

batch_size = 50
dataset_base_path = '/scratch/data/sentiment-classification'
dataset_path_list = [join(dataset_base_path, f) for f in os.listdir(dataset_base_path) 
                                                    if os.path.isdir(join(dataset_base_path, f))]

embedding_flavor_list = list(get_embeddings.embedding_flavor_to_embedding.keys()) 

combinations =  itertools.product(dataset_path_list, embedding_flavor_list, ['train', 'test'])    

commands_to_run = [ 'python get_embeddings.py ' +
                   f'--batch_size {batch_size} '+
                   f'--dataset_path {dataset_path} ' +
                   f'--embedding_flavor {embedding_flavor} ' +
                   f'--train_test {train_test}'
                   for dataset_path, embedding_flavor, train_test in combinations]
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
