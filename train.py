import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.backends.cudnn as cudnn

from train_argument import parser, print_args

from time import time
from data import get_dataloader
from utils import * 
from models import *
from trainer import *


def main(args):
    save_folder = args.affix
    
    log_folder = os.path.join(args.log_root, save_folder)
    model_folder = os.path.join(args.model_root, save_folder)

    makedirs(log_folder)
    makedirs(model_folder)

    setattr(args, 'log_folder', log_folder)
    setattr(args, 'model_folder', model_folder)

    logger = create_logger(log_folder, 'train', 'info')
    print_args(args, logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic=True 
        torch.backends.cudnn.benchmark=False
    
    if args.model == "vgg16":
        net = vgg(depth=16)
        if args.mask:
            net = masked_vgg(depth=16)
    elif args.model == "resnet18":
        net = resnet18()
        if args.mask:
            net = masked_resnet18()
    elif args.model == "lenet300_100":
        net = LeNet_300_100()
        if args.mask:
            net = LeNet_300_100_Masked()
    else:
        if args.mask:
            net = LeNet_5_Masked()
        else:
            net = LeNet_5()

    net.to(device)
    
    trainer = Trainer(args, logger)
    
    loss = nn.CrossEntropyLoss()
 
    
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    train_loader = get_dataloader(args.model, train = True, batch_size = args.batch_size)
    test_loader = get_dataloader(args.model, train = True, batch_size = args.batch_size)
    # if args.dataset == 'cifar10':
    #     train_loader = torch.utils.data.DataLoader(
    #         datasets.CIFAR10('./data.cifar10', train=True, download=True,
    #                     transform=transforms.Compose([
    #                         transforms.Pad(4),
    #                         transforms.RandomCrop(32),
    #                         transforms.RandomHorizontalFlip(),
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #                     ])),
    #         batch_size=args.batch_size, shuffle=True, **kwargs)
    #     test_loader = torch.utils.data.DataLoader(
    #         datasets.CIFAR10('./data.cifar10', train=False, transform=transforms.Compose([
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #                     ])),
    #         batch_size=100, shuffle=True, **kwargs)
    # else:
    #     train_loader = torch.utils.data.DataLoader(
    #         datasets.CIFAR100('./data.cifar100', train=True, download=True,
    #                     transform=transforms.Compose([
    #                         transforms.Pad(4),
    #                         transforms.RandomCrop(32),
    #                         transforms.RandomHorizontalFlip(),
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #                     ])),
    #         batch_size=args.batch_size, shuffle=True, **kwargs)
    #     test_loader = torch.utils.data.DataLoader(
    #         datasets.CIFAR100('./data.cifar100', train=False, transform=transforms.Compose([
    #                         transforms.ToTensor(),
    #                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #                     ])),
    #         batch_size=100, shuffle=True, **kwargs)
        
    
    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 120], gamma=0.1)
    trainer.train(net, loss, device, train_loader, test_loader, optimizer=optimizer, scheduler=scheduler)
    


if __name__ == '__main__':
    args = parser()
    #print_args(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(args)