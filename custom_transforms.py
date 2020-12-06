from torchvision import transforms
import torch
from PIL import Image

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., noise_level=1.):
        self.std = std
        self.mean = mean
        self.noise_level = noise_level

    def __call__(self, tensor):
        new_tensor = tensor + (torch.randn(tensor.size()) * self.std + self.mean) * self.noise_level
        return torch.clamp(new_tensor, min=0, max=1) 
    
    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std}, noise_level={self.noise_level})'

TITRATION_TRANSFORM = lambda noise_level: transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(noise_level=noise_level)
])

ERASE_TRANSFORM = lambda erase_probability: transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomErasing(p=erase_probability, scale=(0.1, 0.4)),
])
