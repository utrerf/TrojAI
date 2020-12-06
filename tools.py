import skimage.io
import re
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.metrics import f1_score
from torchvision import transforms
from PIL import Image
from robustness.datasets import DataSet 

def set_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_model_info(model): 
    model_info = {}
    total_parameters = 0
    for parameter in model.parameters():
        num_parameters = 1
        for dim in parameter.size():
            num_parameters *= dim
        total_parameters += num_parameters
    model_info['total_parameters'] = total_parameters
    model_info['num_classes'] = len(parameter)
    return model_info

class TrojAIDataset(Dataset):
    # Makes the dataset from examples_dirpath
    def __init__(self, examples_dirpath, transform=None):
        self.examples_dirpath = examples_dirpath
        self.transform = transform
        self.x, self.y = [], []
        center_crop_transform = transforms.CenterCrop(224)
        for image_filename in os.listdir(examples_dirpath):
            img = Image.open(os.path.join(examples_dirpath, image_filename))
            img = img.convert('RGB')
            img = center_crop_transform(img)
            self.x.append(img)
        
            label = int(re.findall(r'class_(\d+)_', image_filename)[0])
            self.y.append(label)
        self.y = torch.LongTensor(self.y)
        self.default_transform = transforms.ToTensor()

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        if self.transform: 
            x = self.transform(x)
        else:
            x = self.default_transform(x)
        return x, y

    def __len__(self):
        return len(self.x)


def get_f1_score(model, loader, num_iterations):
    # compute the mean f1 score over num_iterations
    score_total = 0
    for iteration in range(num_iterations):
        preds   = np.zeros(len(loader.dataset))
        targets = np.zeros(len(loader.dataset))
        current_idx = 0
        for inp, target in loader:
            with torch.no_grad():
                inp = inp.cuda()
                output = model(inp)
                n = len(target)
                argmax =  torch.argmax(output, dim=1).type(torch.uint8).cpu().numpy().flatten()
                preds[current_idx : current_idx+n] += argmax
                targets[current_idx : current_idx+n] += target.numpy()
                current_idx += n
        score_total += f1_score(targets, preds, average='macro')
        
    return score_total / num_iterations


def get_transform_score(model, dataset, transform, 
                        num_iterations=5, num_workers=0, batch_size=128):
    dataset.transform = transform
    loader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=True, num_workers=num_workers, pin_memory=True)
    return get_f1_score(model, loader, num_iterations=num_iterations)


class MadryDataset(DataSet):
    def __init__(self, data_path, **kwargs):
        ds_name = 'madry_dataset'
        ds_kwargs = {
            'num_classes': kwargs['num_classes'],
            'mean': torch.tensor([0., 0., 0.]).cuda(),
            'std': torch.tensor([1., 1., 1.]).cuda(),
            'custom_class': None,
            'label_mapping': None,
            'transform_train': None,
            'transform_test': None,
            }
        ds_kwargs = self.override_args(ds_kwargs, kwargs)
        super(MadryDataset, self).__init__(ds_name,
                                           data_path, **ds_kwargs)
class CustomAdvModel(torch.nn.Module):
    def __init__(self, model):
        super(CustomAdvModel, self).__init__()
        self.model = model
    def forward(self, inp, with_latent=False,
            fake_relu=False, no_relu=False):
        output = self.model(inp)
        return output

def get_adv_score(model, dataset, constraint='inf', eps=0.04, iterations=3, 
                  transform=None, batch_size=16, num_workers=0, num_iterations=1):
    attack_kwargs = {
            'constraint': constraint, # L-inf PGD
            'eps': eps, # Epsilon constraint (L-inf norm)
            'step_size': 2.5*(eps/iterations), # Learning rate for PGD
            'iterations': iterations, # Number of PGD steps
            'targeted': False, # Targeted attack
            'custom_loss': None, # Use default cross-entropy loss
            'random_start': True
            }
    dataset.transform = transform
    loader = DataLoader(dataset, batch_size=batch_size, 
                        shuffle=True, num_workers=num_workers, pin_memory=True)
    return get_adv_f1_score(model, loader, num_iterations, attack_kwargs)

def get_adv_f1_score(model, loader, num_iterations, attack_kwargs):
    # compute the mean f1 score over num_iterations
    score_total = 0
    for iteration in range(num_iterations):
        preds   = np.zeros(len(loader.dataset))
        targets = np.zeros(len(loader.dataset))
        current_idx = 0
        for inp, target in loader:
            inp = inp.cuda()
            output, im_adv = model(inp, target.cuda(), make_adv=True, **attack_kwargs)
            with torch.no_grad():
                n = len(target)
                argmax =  torch.argmax(output, dim=1).cpu().numpy().flatten()
                preds[current_idx : current_idx+n] += argmax
                targets[current_idx : current_idx+n] += target.numpy()
                current_idx += n
        score_total += f1_score(targets, preds, average='macro')
        
    return score_total / num_iterations

def extract_scores(dir_name):
    df = pd.DataFrame([])
    for i, filename in enumerate(os.listdir(dir_name)):
        full_path = os.path.join(dir_name, filename)
        model_id = filename[:-4]
        if i == 0:
            df = pd.read_csv(full_path)
            df['model_name'] = model_id
        else:
            new_df = pd.read_csv(full_path)
            new_df['model_name'] = model_id
            df = df.append(new_df)
    return df

