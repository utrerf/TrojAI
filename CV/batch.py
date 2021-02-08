import os
import time
import subprocess
import shlex

# gpu_list = list(range(4,8))
gpu_list = [1,2,3,4,5,6,7]
gpus_per_command = 1
polling_delay_seconds = 1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

def id_gen(i):
    return str(100000000 + i)[1:]

base_dir = '/scratch/utrerf/round4/models'

commands_to_run = [ 'python trojan_detector.py ' +
                   f'--model_filepath {base_dir}/id-{id_gen(i)}/model.pt '+
                   f'--examples_dirpath {base_dir}/id-{id_gen(i)}/clean_example_data ' +
                   f'--is_train True '
                   for i in range(0, 1050)]
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
            

