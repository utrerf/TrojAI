import os
import numpy as np
import copy
import transformers
import jsonpickle
import argparse
import json
import model_factories
import torch

import warnings
warnings.filterwarnings("ignore")


def trojan_detector(model_filepath, cls_token_is_first, tokenizer_filepath, embedding_filepath, 
                                            result_filepath, scratch_dirpath, examples_dirpath):

    use_amp = True  # attempt to use mixed precision to accelerate embedding conversion process
    
    # 1. Load the model, examples, tokenizer, and embedding model
    model = torch.load(model_filepath, map_location=torch.device('cuda'))

    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
    fns.sort()  # ensure file ordering
    if len(fns) > 5: fns = fns[0:5]  # limit to 5 examples
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
 
    # TODO this uses the correct huggingface tokenizer instead of the one provided by the filepath, since GitHub has a 100MB file size limit
    # tokenizer = torch.load(tokenizer_filepath)
    tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # load the specified embedding
    # embedding = torch.load(embedding_filepath, map_location=torch.device(device))
    embedding = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

    # identify the max sequence length for the given embedding
    max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]
    
    # 2. Do inference on each example
    for fn in fns:
        # load the example
        with open(fn, 'r') as fh:
            text = fh.readline()

        # tokenize the text
        results = tokenizer(text, max_length=max_input_length - 2, padding=True, truncation=True, return_tensors="pt")
        # extract the input token ids and the attention mask
        input_ids = results.data['input_ids']
        attention_mask = results.data['attention_mask']

        # convert to embedding
        with torch.no_grad():
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            if use_amp:
                with torch.cuda.amp.autocast():
                    embedding_vector = embedding(input_ids, attention_mask=attention_mask)[0]
            else:
                embedding_vector = embedding(input_ids, attention_mask=attention_mask)[0]

            if cls_token_is_first:
                embedding_vector = embedding_vector[:, 0, :]
            else:
                embedding_vector = embedding_vector[:, -1, :]

            embedding_vector = embedding_vector.to('cpu')
            embedding_vector = embedding_vector.numpy()

            # reshape embedding vector to create batch size of 1
            embedding_vector = np.expand_dims(embedding_vector, axis=0)
            # embedding_vector is [1, 1, <embedding length>]
            adv_embedding_vector = copy.deepcopy(embedding_vector)

        embedding_vector = torch.from_numpy(embedding_vector).to(device)
        # predict the text sentiment
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(embedding_vector).cpu().detach().numpy()
        else:
            logits = model(embedding_vector).cpu().detach().numpy()

        sentiment_pred = np.argmax(logits)
        print('Sentiment: {} from Text: "{}"'.format(sentiment_pred, text))
        print('  logits: {}'.format(logits))


        # create a prediction tensor without graph connections by copying it to a numpy array
        pred_tensor = torch.from_numpy(np.asarray(sentiment_pred)).reshape(-1).to(device)
        # predicted sentiment stands if for the ground truth label
        y_truth = pred_tensor
        adv_embedding_vector = torch.from_numpy(adv_embedding_vector).to(device)


    # Test scratch space
    with open(os.path.join(scratch_dirpath, 'test.txt'), 'w') as fh:
        fh.write('this is a test')

    # TODO: Get a classifier to compute the trojan probability
    trojan_probability = np.random.rand()
    print('Trojan Probability: {}'.format(trojan_probability))

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_filepath', type=str, 
                        help='File path to the pytorch model file to be evaluated.', 
                        default='./model/model.pt')
    parser.add_argument('--cls_token_is_first', type=bool, 
                        help='Whether the first embedding token should be used as the'+
                             ' summary of the text sequence, or the last token.', 
                        default=True)
    parser.add_argument('--tokenizer_filepath', type=str, 
                        help='File path to the pytorch model file to be evaluated.', 
                        default='./model/tokenizer.pt')
    parser.add_argument('--embedding_filepath', type=str, 
                        help='File path to the pytorch model file to be evaluated.', 
                        default='./model/embedding.pt')
    parser.add_argument('--result_filepath', type=str, 
                        help='File path to the file where output result should be written.'+
                             ' After execution this file should contain a single line with'+
                                             ' a single floating point trojan probability.', 
                        default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, 
                         help='File path to the folder where scratch disk space exists. '+
                              'This folder will be empty at execution start and will be '+ 
                              'deleted at completion of execution.', 
                         default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, 
                         help='File path to the folder of examples which might be useful'+
                              ' for determining whether a model is poisoned.', 
                         default='./model/clean_example_data')
    parser.add_argument('--is_train', type=bool, 
                         help='If true, we save features. Otherwise, we make a prediction',
                         default=False)
    
    args = parser.parse_args()
    print(args)
    #for k, v in args.items():
    #    print(f'{k}: {v}')

    trojan_detector(args.model_filepath, args.cls_token_is_first, args.tokenizer_filepath, 
                    args.embedding_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath)

