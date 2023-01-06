import torch
from torchvision import datasets 
from torchvision.transforms import ToTensor, transforms



def get_dataloader(model, train, batch_size):
    if model == "vgg16":
        loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root = './data.cifar10', train=train, download=True,
                    transform=transforms.Compose([
                        transforms.Pad(4),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
        batch_size=batch_size, shuffle=True,)
    
    elif model == "resnet18":
        loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root = './data.cifar100', train=train, download=True,
                    transform=transforms.Compose([
                        transforms.Pad(4),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                    ])),
        batch_size=batch_size, shuffle=True, )
    
    else:
        loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='./data', train=train, transform=ToTensor(), download=True), 
        batch_size=batch_size, shuffle=True)

    return loader