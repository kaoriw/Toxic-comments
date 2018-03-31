rom __future__ import print_function
import os
import os.path
import sys
import random
import time

import numpy as np
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torch.autograd
import torch.autograd.variable
import torchvision
import torchvision.transforms

#Embedding is done
#We have a vector of 2 dimensions: one for each word, one for the list of same-context words
#x contains the embedding results


FEATURE = 100


lstm = nn.LSTM(input_size=FEATURE, hidden_size=60, num_layers=2)
classifier = nn.Linear(32,2, bias=True)
lstm.train()
classifier.train()

losslayer = nn.CrossEntropyLoss()

batchsize = 128

class Lstm(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=FEATURE, hidden_size=60, num_layers=2)
        self.classifier = nn.Linear(32,2, bias=True)
    def forward(self, x):
        x = nn.MaxPool1d(10)(lstm(x))
        x = nn.Dropout(inplace = True)(classifier(x))
