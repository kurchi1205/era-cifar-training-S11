from utils import CIFAR10WithAlbumentations, train_transforms, test_transforms
from models.resnet18 import Resnet18
import torch
from torchsummary import summary

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def get_data():
    train_ds = CIFAR10WithAlbumentations('./data', train=True, download=True, transform=train_transforms)
    test_ds = CIFAR10WithAlbumentations('./data', train=False, download=True, transform=test_transforms)
    return train_ds, test_ds


def get_dataloader(data, shuffle=True, num_workers=4):
    dataloader_args = dict(shuffle=shuffle, batch_size=512, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    loader = torch.utils.data.DataLoader(data, **dataloader_args)
    return loader

def get_model_summary():
    model = Resnet18().to(device)
    summary(model, input_size=(3, 32, 32))