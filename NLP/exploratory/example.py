# The data exists in a json file that can be loaded like:
with open(self.data_json_filepath, mode='r', encoding='utf-8') as f:
    json_data = jsonpickle.decode(f.read())

# once you have selected a sentence to inference, you need to define the embedding using information in the config.json file
config.embedding = # This comes from the config.json file
config.embedding_flavor = # This comes from the config.json file
text_data = # this is the text string you want to classisfy as positive or negative
tokenizer = None
embedding = None
if config.embedding == 'BERT':
    tokenizer = transformers.BertTokenizer.from_pretrained(config.embedding_flavor)
    embedding = transformers.BertModel.from_pretrained(config.embedding_flavor)
elif config.embedding == 'GPT-2':
    # ignore missing weights warning
    # https://github.com/huggingface/transformers/issues/5800
    # https://github.com/huggingface/transformers/pull/5922
    tokenizer = transformers.GPT2Tokenizer.from_pretrained(config.embedding_flavor)
    embedding = transformers.GPT2Model.from_pretrained(config.embedding_flavor)
elif config.embedding == 'DistilBERT':
    tokenizer = transformers.DistilBertTokenizer.from_pretrained(config.embedding_flavor)
    embedding = transformers.DistilBertModel.from_pretrained(config.embedding_flavor)
else:
    raise RuntimeError('Invalid Embedding Type: {}'.format(config.embedding))

# move the embedding to the GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding.to(device)
pad_size = max_input_length - 2
if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
logger.info('Converting text representation to embedding')
results = tokenizer(text, padding='max_length', max_length=pad_size, truncation=True, return_tensors="pt")
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
    embedding_vector = embedding_vector.to('cpu').squeeze()

# at this point you can pass the embedding_vector tensor through the model you load from my dataset
model = torch.load(model_filepath, map_location=torch.device(device))
embedding_vector = embedding_vector.to(device)
pred = model(embedding_vector)



# Here is the LstmLinear model definition
class LstmLinearModel(torch.nn.Module):
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, 
                 dropout: float, bidirectional: bool, n_layers: int):
        super().__init__()
        self.rnn = torch.nn.LSTM(input_size,
                          hidden_size,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)
        self.linear = torch.nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, data):
        # input data is after the embedding
        # data = [batch size, sent len, emb dim]
        _, hidden = self.rnn(data)
        # hidden = [n layers * n directions, batch size, emb dim]
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])
        # hidden = [batch size, hid dim]
        output = self.linear(hidden)
        # output = [batch size, out dim]
        return output

