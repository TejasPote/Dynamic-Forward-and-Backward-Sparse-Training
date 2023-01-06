import torch
import numpy as np
import os
import sys
import time
import math
import json
import logging

import torch.nn as nn
import torch.nn.init as init

import numpy as np
from models import MaskedMLP, MaskedConv2d




def numpy2cuda(array):
    tensor = torch.from_numpy(array)

def list2cuda(_list):
    array = np.array(_list)
    return numpy2cuda(array)

def tensor2cuda(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()

    return tensor

def evaluate(_input, _target, method='mean'):
    correct = (_input == _target).astype(np.float32)
    if method == 'mean':
        return correct.mean()
    else:
        return correct.sum()

def create_logger(save_path='', file_type='', level='debug'):

    if level == 'debug':
        _level = logging.DEBUG
    elif level == 'info':
        _level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(_level)

    cs = logging.StreamHandler()
    cs.setLevel(_level)
    logger.addHandler(cs)

    if save_path != '':
        file_name = os.path.join(save_path, file_type + '_log.txt')
        fh = logging.FileHandler(file_name, mode='w')
        fh.setLevel(_level)

        logger.addHandler(fh)

    return logger


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)





def print_layer_keep_ratio(model):
    total = 0. 
    keep = 0.
    for layer in model.modules():
      abs_weight = torch.abs(layer.weight)
      threshold = layer.threshold.view(abs_weight.shape[0], -1)
      abs_weight = abs_weight-threshold
      mask = layer.step(abs_weight)
      ratio = torch.sum(mask) / mask.numel()
      total += mask.numel()
      keep += torch.sum(mask)
      logger.info("{}, keep ratio {:.4f}".format(layer, ratio))
    return keep/total