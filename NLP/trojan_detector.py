import transformers
import jsonpickle
import argparse
import json
import model_factories
import torch

def trojan_detector(args):
    # load the entire dataset
    # with open(args.data_json_filepath, mode='r', encoding='utf-8') as f:
    #     json_data = jsonpickle.decode(f.read())
    text = "The story centers around Barry McKenzie who must go to England if he wishes to claim his inheritance. Being about the grossest Aussie shearer ever to set foot outside this great Nation of ours there is something of a culture clash and much fun and games ensue. The songs of Barry McKenzie(Barry Crocker) are highlights."
    config = None
    with open(args.config_filepath, mode='r', encoding='utf-8') as f:
        config = jsonpickle.decode(f.read())
    tokenizer = None
    embedding = None
    if config['embedding'] == 'BERT':
        tokenizer = transformers.BertTokenizer.from_pretrained(config['embedding_flavor'])
        embedding = transformers.BertModel.from_pretrained(config['embedding_flavor'])
    elif config['embedding']== 'GPT-2':
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(config['embedding_flavor'])
        embedding = transformers.GPT2Model.from_pretrained(config['embedding_flavor'])
    elif config['embedding'] == 'DistilBERT':
        tokenizer = transformers.DistilBertTokenizer.from_pretrained(config['embedding_flavor'])
        embedding = transformers.DistilBertModel.from_pretrained(config['embedding_flavor'])
    else:
        raise RuntimeError('Invalid Embedding Type: {}'.format(config.embedding))

    # move the embedding to the GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding.to(device)
    max_input_length = 1000
    pad_size = max_input_length - 2
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # results = tokenizer(text, padding='max_length', max_length=pad_size, truncation=True, return_tensors="pt")
    results = tokenizer(text, padding='max_length', truncation=True, return_tensors="pt")
    input_ids = results.data['input_ids']
    attention_mask = results.data['attention_mask']
    
    print(f'input_ids: {input_ids}')
    print(f'input_ids: {attention_mask}')

    # convert to embedding
    with torch.no_grad():
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
    embedding_vector = embedding(input_ids, attention_mask=attention_mask)
    embedding_vector = embedding(input_ids, attention_mask=attention_mask).last_hidden_state
    embedding_vector = embedding_vector.to('cpu').squeeze()[0].reshape(1,1,-1)
    
    # at this point you can pass the embedding_vector tensor through the model you load from my dataset
    model = torch.load(args.model_filepath, map_location=torch.device(device))
    embedding_vector = embedding_vector.to(device)
    pred = model(embedding_vector)



if __name__ == '__main__':
    default_path = '/scratch/utrerf/round5-initial/models'
    default_model = 'id-n0000000'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_filepath', type=str, 
                        help='File path with the config file for the model.', 
                        default=f'{default_path}/{default_model}/config.json')
    parser.add_argument('--model_filepath', type=str, 
                        help='File path to the pytorch model file to be evaluated.', 
                        default=f'{default_path}/{default_model}/model.pt')
    parser.add_argument('--data_filepath', type=str, 
                        help='File path to the pytorch model file to be evaluated.', 
                        default='/scratch/data')
    args = parser.parse_args()
    # data_json_filepath
    trojan_detector(args)
