
import copy

import torch
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader 
from torch import cuda, FloatTensor
from torch import nn
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

from models import *
from utils import *


############
## CONFIG ##
############

class Config:
  def __init__(self):
    self.epochs = 30
    self.batch_size = 20
    self.lr = 0.06
    self.nt = 54
    self.nv = 9
    self.hidden_size = 100
    self.feats = "features.json"
    self.labels = "labels.json"
    self.max_length = 300
    self.use_gpu = False
    self.dtype = cuda.FloatTensor if self.use_gpu else FloatTensor
    self.num_classes = 2

  def __str__(self):
    properties = vars(self)
    properties = ["{} : {}".format(k, str(v)) for k, v in properties.items()]
    properties = '\n'.join(properties)
    properties = "--- Config --- \n" + properties + "\n"
    return properties

############
# TRAINING #
############

def train(model, loss_function, optimizer, num_epochs = 1, logger = None):
  best_model = None
  best_val_acc = 0
  for epoch in range(num_epochs):
      print('Starting epoch %d / %d' % (epoch + 1, num_epochs))
      model.train()
      loss_total = 0
      for t, (x, y, _) in enumerate(model.config.train_loader):
          x_var = Variable(x)
          y_var = Variable(y.type(model.config.dtype).long())
          scores = model(x_var) 
          loss = loss_function(scores, y_var)
          loss_total += loss.data[0]
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          if ((t+1) % 10) == 0:
            grad_magnitude = [(x.grad.data.sum(), torch.numel(x.grad.data)) for x in model.parameters() if x.grad.data.sum() != 0.0]
            grad_magnitude = sum([abs(x[0]) for x in grad_magnitude]) #/ sum([x[1] for x in grad_magnitude])
      check_accuracy(model, model.config.train_loader, type = "train")
      test_acc = check_accuracy(model, model.config.test_loader, type = "test")
      if test_acc > best_val_acc:
        best_val_acc = test_acc
        best_model = copy.deepcopy(model)
  return best_model


def check_accuracy(model, loader, type="", logger = None):
  num_correct = 0
  num_samples = 0
  examples, all_labels, all_predicted = [], [], []
  model.eval()
  for t, (x, y, keys) in enumerate(loader):
      x_var = Variable(x)
      scores = model(x_var)
      _, preds = scores.data.cpu().max(1)
      num_correct += (preds == y).sum()
      num_samples += preds.size(0)
      examples.extend(keys)
      all_labels.extend(list(y))
      all_predicted.extend(list(np.ndarray.flatten(preds.numpy())))
  acc = float(num_correct) / num_samples
  return acc
 
def final_check_accuracy(model, loader, type="", logger = None):
  print("Checking accuracy on {} set".format(type))
  num_correct = 0
  num_samples = 0
  examples, all_labels, all_predicted = [], [], []
  model.eval() # Put the model in test mode (the opposite of model.train(), essentially)
  for t, (x, y, keys) in enumerate(loader):
      x_var = Variable(x)
      scores = model(x_var)
      _, preds = scores.data.cpu().max(1)
      num_correct += (preds == y).sum()
      num_samples += preds.size(0)
      examples.extend(keys)
      all_labels.extend(list(y))
      all_predicted.extend(list(np.ndarray.flatten(preds.numpy())))
  acc = float(num_correct) / num_samples
  print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
  if logger:
    data = "type,file,predicted,label\n"
    for i in range(len(examples)):
      data += "{},{},{},{}\n".format(type, examples[i], all_predicted[i], all_labels[i])
    logger.logResult(data)
  return acc

########
# MAIN #
########

def main():
  # Config
  config = Config() 
  print(config)

  logger = Logger()

  # Model
  model = SimpleAudioRNN(config)
  model.apply(initialize_weights)
  if config.use_gpu:
    model = model.cuda()

  # Load Data
  dataset = getAudioDatasets(config)
  
  train_idx, test_idx = splitIndices(dataset, config.nt, config.nv, shuffle = True)
  train_sampler, test_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(test_idx)

  train_loader = DataLoader(dataset, batch_size = config.batch_size, num_workers = 3, sampler = train_sampler)
  test_loader = DataLoader(dataset, batch_size = config.batch_size, num_workers = 1, sampler = test_sampler)

  config.train_loader = train_loader
  config.test_loader = test_loader

  optimizer = optim.Adam(model.parameters(), lr=config.lr) 
  loss_function = nn.CrossEntropyLoss().type(config.dtype)
  best_model = train(model, loss_function, optimizer, config.epochs, logger=logger)

  print("\n--- Final Evaluation ---")
  final_check_accuracy(best_model, best_model.config.train_loader, type = "train", logger = logger)
  final_check_accuracy(best_model, best_model.config.test_loader, type = "test", logger = logger)


if __name__ == '__main__':
  main()
