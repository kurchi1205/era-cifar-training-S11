from utils import CIFAR10WithAlbumentations, train_transforms, test_transforms, get_misclassified_images_with_label
from models.resnet18 import Resnet18
from models.resnet import ResNet18
import torch
import torch.nn as nn
from torchsummary import summary
from torch_lr_finder import LRFinder
from tqdm import tqdm

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")


def get_data():
    train_ds = CIFAR10WithAlbumentations('./data', train=True, download=True, transform=train_transforms)
    test_ds = CIFAR10WithAlbumentations('./data', train=False, download=True, transform=test_transforms)
    return train_ds, test_ds


def get_dataloader(data, batch_size, shuffle=True, num_workers=4):
    dataloader_args = dict(shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
    loader = torch.utils.data.DataLoader(data, **dataloader_args)
    return loader


def get_model():
    model = ResNet18().to(device)
    return model

def get_model_summary(model):
    summary(model, input_size=(3, 32, 32))


def get_optimizer(model, lr=0.01, optimizer_type = "adam"):
    if optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    return optimizer


def get_lr(model, optimizer, train_loader, test_loader, num_iter):
    criterion = nn.CrossEntropyLoss()
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, val_loader=test_loader, end_lr=1, num_iter=num_iter, step_mode="exp")
    lr = lr_finder.plot(log_lr=True, suggest_lr=True)
    lr_finder.reset()


def get_scheduler(train_loader, optimizer, epochs, max_lr, div_factor=10, final_div_factor=100, three_phase=True):
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, epochs=epochs, steps_per_epoch=len(train_loader), anneal_strategy='cos',\
                                                div_factor=div_factor, final_div_factor=final_div_factor, three_phase=three_phase)
    return scheduler


def train(model, device, train_loader, optimizer, scheduler, epoch, train_losses, train_acc):
      model.train()
      pbar = tqdm(train_loader)
      correct = 0
      processed = 0
      loss_fn = nn.CrossEntropyLoss()
      for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes.
        # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = loss_fn(y_pred, target)
        train_losses.append(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
        train_acc.append(100*correct/processed)


def test(model, device, test_loader, test_losses, test_acc):
    model.eval()
    test_loss = 0
    correct = 0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
                
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    test_acc.append(100. * correct / len(test_loader.dataset))


def train_model(epochs, model, train_loader, test_loader, optimizer, scheduler=None):
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    for epoch in range(epochs):
        print("EPOCH:", epoch)
        train(model, device, train_loader, optimizer, scheduler, epoch, train_losses, train_acc)
        test(model, device, test_loader, test_losses, test_acc)


def infer(model, device, infer_loader, misclassified):
    model.eval()
    with torch.no_grad():
        for data, target in infer_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            print(pred)
            print(target)
            print(pred.eq(target.view_as(pred)).sum().item())
            if pred.eq(target.view_as(pred)).sum().item() == 0:
                misclassified.append(get_misclassified_images_with_label(data, pred))
            if len(misclassified) == 10:
                break