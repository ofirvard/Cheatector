import torch
import ujson
from torch.utils.data.dataset import Dataset
import random
import numpy as np
from torch.autograd import Variable
import re
import json
from datetime import datetime
import os
import math


##########
# Logger #
#########

logDir = "./Results"

class Logger():
  def __init__(self, title="results"):
    self.timestamp = datetime.now().strftime("%d.%m_%H:%M:%S")
    if not os.path.isdir(logDir):
	os.makedirs(logDir)
    self.save_path = logDir
    self.results = os.path.join(self.save_path, (self.timestamp + ".csv"))

  def logResult(self, data):
    with open(self.results, 'a+') as f:
	f.write(data)

  def __str__(self):
    return self.save_path

##############
# CONSTANTS #
##############

LABELS = { "lie": 1, "truth": 0 }

####################
# DATASET HELPERS #
####################

def pad_tensor(tensor, length):
  D, T = tensor.size()
  zeros = torch.FloatTensor(D, length - T).zero_()
  new = torch.cat((tensor, zeros), dim = 1)
  return new

def splitIndices(dataset, num_val):
    length = len(dataset)
    indices = list(range(0, length))
    random.shuffle(indices)
    train = indices[0:num_val]
    val = indices[num_val:]
    return train, val
    
def splitIndices(dataset):
    length = len(dataset)
    indices = list(range(0, length))
    return indices, indices

def splitIndicesleftOne(dataset, i):
    length = len(dataset)
    indices = list(range(0, length))
    train = indices[0:i] + indices[i+1:] 
    val = indices[i:i+1]
    return train, val

"""
returns two AudioDatasets: one that contains all subjects except those specified to be held out for testing, and one that contains only the held out subjects.
If no heldout subjects are specified, then simply returns all the data in one dataset and returns None instead of a test dataset.
"""
def getAudioDatasets(config):
    return AudioDataset(config)

#############
# DATASETS #
#############

class AudioDataset(Dataset):
    """Dataset wrapping data and target tensors. Naive implementation does data preprocessing per 'get_item' call
    Each sample will be retrieved by indexing both tensors along the first
    dimension.
    
    Arguments:
        data_path (str): path to image folder
    """
    def __init__(self, config):
      self.labels, self.features, self.examples = None, None, None
      with open(config.feats, 'r') as data:
        self.features = json.load(data)
      with open(config.labels, 'r') as labels:
        self.labels = json.load(labels)
      assert(self.features and self.labels)
      
      self.num_examples = len(self.labels.keys())
      self.examples = [k for k in self.labels.keys()]
        
      for file in self.examples:
        feat_tensor = (torch.FloatTensor(self.features[file])[:, 0:config.max_length]).contiguous().type(config.dtype)
        if feat_tensor.size(1) < config.max_length:
            feat_tensor = pad_tensor(feat_tensor, config.max_length)
        del self.features[file]
        self.features[file] = feat_tensor.t()

    def __getitem__(self, idx):
      key = self.examples[idx]
      label = self.labels[key]
      feats = self.features[key]
      return feats, label, key

    def __len__(self):
      return self.num_examples
