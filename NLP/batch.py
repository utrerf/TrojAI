import os
import time
import subprocess
import shlex
import pandas as pd

# gpu_list = list(range(4,8))
gpu_list = [4,5,6,7]
gpus_per_command = 1
polling_delay_seconds = 1
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# CONSTANTS
embedding_flavor_to_tokenizer = {'gpt2':"GPT-2-gpt2.pt",
                                 'distilbert-base-uncased':"DistilBERT-distilbert-base-uncased.pt",
                                 'bert-base-uncased':"BERT-bert-base-uncased.pt"}

def get_metadata_file(base_path = "/scratch/data/round5-train-dataset"):
    metadata = pd.read_csv(os.path.join(base_path, "METADATA.csv"))
    return metadata

def get_command_list(metadata):
    commands_df = metadata[['model_name', 'poisoned', 'embedding_flavor', 'source_dataset', 'cls_token_is_first']]
    commands_df['tokenizer_filename'] = commands_df['embedding_flavor'].map(embedding_flavor_to_tokenizer)
    commands_df['embedding_filename'] = commands_df['embedding_flavor'].map(embedding_flavor_to_tokenizer)
    commands_df.head()

    model_training_data_path = '/scratch/data/round5-train-dataset/'
    text_data_path = '/scratch/data/sentiment-classification/'


    commands = [f'python trojan_detector.py ' +
     f'--model_filepath {os.path.join(model_training_data_path, "models", commands_df.loc[[i]].model_name.item(), "model.pt")} ' +
     f'--cls_token_is_first {commands_df.loc[[i]].cls_token_is_first.item()} ' +
     f'--tokenizer_filepath {os.path.join(model_training_data_path, "tokenizers", commands_df.loc[[i]].tokenizer_filename.item())} ' + 
     f'--embedding_filepath {os.path.join(model_training_data_path, "embeddings", commands_df.loc[[i]].embedding_filename.item())} ' +
     f'--examples_dirpath {os.path.join(model_training_data_path, "models", commands_df.loc[[i]].model_name.item(), "clean_example_data")} ' + 
     f'--train_test train'
               for i in range(len(commands_df))]
    commands.reverse()
    return commands

commands_to_run = get_command_list(get_metadata_file())

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
            

