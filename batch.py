import os
import time
import subprocess
import shlex

gpu_list = list(range(4))
gpus_per_command = 1
polling_delay_seconds = 1

def id_gen(i):
    return str(100000000 + i)[1:]

commands_to_run = [ 'python trojan_detector.py ' +
                   f'--model_filepath /home/ubuntu/round2/models/id-{id_gen(i)}/model.pt '+
                   f'--examples_dirpath /home/ubuntu/round2/models/id-{id_gen(i)}/example_data ' 
                   for i in range(1000)]
commands_to_run.reverse()

def poll_process(process):
    time.sleep(polling_delay_seconds)
    return process.poll()

pid_to_process_and_gpus = {}
free_gpus = set(gpu_list)
while len(commands_to_run) > 0:
    
    # try kicking off process if we have free gpus
    while len(free_gpus) >= gpus_per_command:
        print(free_gpus)
        gpus = []
        for i in range(gpus_per_command):
            # updates free_gpus
            gpus.append(str(free_gpus.pop()))
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(gpus)
        command = commands_to_run.pop()
        subproc = subprocess.Popen(shlex.split(command))
        # updates_dict
        pid_to_process_and_gpus[subproc.pid] = (subproc, gpus)
    
    # update free_gpus
    for pid, (current_process, gpus) in pid_to_process_and_gpus.copy().items():
        if poll_process(current_process) is not None:
            print(f'done with {pid}')
            free_gpus.update(gpus)
            del pid_to_process_and_gpus[pid]
            

