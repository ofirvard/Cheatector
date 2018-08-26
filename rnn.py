#!/usr/bin/python
# -*- coding: utf-8 -*-

import copy
import argparse

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
    def __init__(self, args):
        self.epochs = 10
        self.batch_size = 20
        self.lr = 0.05
        self.hidden_size = 100
        self.feats = 'features.json'
        self.labels = 'labels.json'
        self.max_length = 300
        self.use_gpu = False
        self.dtype = (cuda.FloatTensor if self.use_gpu else FloatTensor)
        self.num_classes = 2
	self.model = args.model
        self.left_one = True

    def __str__(self):
        properties = vars(self)
        properties = ['{} : {}'.format(k, str(v)) for (k, v) in properties.items()]
        properties = '\n'.join(properties)
        properties = '--- Config --- \n' + properties + '\n'
        return properties

def parseConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default = None)
    return parser.parse_args()


############
# TRAINING #
############

def train(model, loss_function, optimizer, num_epochs=1):
    best_model = None
    best_val_acc = 0
    for epoch in range(num_epochs):
        model.train()
        loss_total = 0
        for (t, (x, y, _)) in enumerate(model.config.train_loader):
            x_var = Variable(x)
            y_var = Variable(y.type(model.config.dtype).long())
            scores = model(x_var)
            loss = loss_function(scores, y_var)
            loss_total += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (t + 1) % 10 == 0:
                grad_magnitude = [(x.grad.data.sum(), torch.numel(x.grad.data)) for x in model.parameters() if x.grad.data.sum() != 0.0]
                grad_magnitude = sum([abs(x[0]) for x in grad_magnitude])

        (truth_correct, truth_wrong, lie_correct, lie_wrong, test_acc) = check_accuracy(model, model.config.train_loader, type='train', toPrint = False)
        if test_acc > best_val_acc:
            best_val_acc = test_acc
            best_model = copy.deepcopy(model)

    return best_model


def check_accuracy(model, loader, type='', toPrint=True):
    num_correct = 0
    num_samples = 0
    (examples, all_labels, all_predicted) = ([], [], [])
    model.eval()
    for (t, (x, y, keys)) in enumerate(loader):
        x_var = Variable(x)
        scores = model(x_var)
        (_, preds) = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += preds.size(0)
        examples.extend(keys)
        all_labels.extend(list(y))
        all_predicted.extend(list(np.ndarray.flatten(preds.numpy())))
    acc = float(num_correct) / num_samples
    truth_correct = 0
    lie_correct = 0
    truth_wrong = 0    
    lie_wrong = 0
    for i in range(len(examples)):
	if(all_labels[i] == 0):
		if(all_predicted[i] == all_labels[i]):
			truth_correct += 1
		else:
			truth_wrong += 1
	else:
		if(all_predicted[i] == all_labels[i]):
			lie_correct += 1
		else:
			lie_wrong += 1
    if (toPrint):
    	print 'truth_correct: {}, truth_wrong: {}, lie_correct: {}, lie_wrong: {}, acc: {}\n'.format(truth_correct, truth_wrong, lie_correct, lie_wrong, acc)    
    return (truth_correct, truth_wrong, lie_correct, lie_wrong, acc) 


########
# MAIN #
########

def main():

	# Config
    args = parseConfig()
    config = Config(args)
    dataset = getAudioDatasets(config)

	# Model
    model = AudioRNN(config)
    if config.model:
	model.load_state_dict(torch.load("./Models/{}".format(config.model)))
        if config.left_one:
	    return
        else:
	    (train_idx, test_idx) = splitIndices(dataset)
	    train_sampler, test_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(test_idx)
 
	    train_loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=3, sampler=train_sampler)
	    test_loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=1, sampler=test_sampler)

	    config.train_loader = train_loader
	    config.test_loader = test_loader

	    acc = check_accuracy(model, model.config.test_loader, type='test', toPrint=True)

	    print '{}'.format(acc)
        return

    else:	
    	model.apply(initialize_weights)
    
	#Train
    if config.left_one:
	    num_correct = 0
	    for i in range(len(dataset)):
		(train_idx, test_idx) = splitIndicesleftOne(dataset, i)
		train_sampler, test_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(test_idx)

		train_loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=3, sampler=train_sampler)
		test_loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=1, sampler=test_sampler)

		config.train_loader = train_loader
		config.test_loader = test_loader

		optimizer = optim.Adam(model.parameters(), lr=config.lr)
		loss_function = nn.CrossEntropyLoss().type(config.dtype)
		best_model = train(model, loss_function, optimizer, config.epochs)

		(truth_correct, truth_wrong, lie_correct, lie_wrong, acc) = check_accuracy(best_model, best_model.config.test_loader, type='test', toPrint=True)
		num_correct += int(acc)

	    print("Score {}".format((num_correct/float(len(dataset)))))

    else:	
	    (train_idx, test_idx) = splitIndices(dataset)
	    train_sampler, test_sampler = SubsetRandomSampler(train_idx), SubsetRandomSampler(test_idx)

	    train_loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=3, sampler=train_sampler)
	    test_loader = DataLoader(dataset, batch_size=config.batch_size, num_workers=1, sampler=test_sampler)

	    config.train_loader = train_loader
	    config.test_loader = test_loader
	
	    optimizer = optim.Adam(model.parameters(), lr=config.lr)
	    loss_function = nn.CrossEntropyLoss().type(config.dtype)
	    best_model = train(model, loss_function, optimizer, config.epochs)

	    (truth_correct, truth_wrong, lie_correct, lie_wrong, acc) = check_accuracy(best_model, best_model.config.test_loader, type='test', toPrint=False)
	    print 'BEST MODEL = truth_correct: {}, truth_wrong: {}, lie_correct: {}, lie_wrong: {}, acc: {}\n'.format(truth_correct, truth_wrong, lie_correct, lie_wrong, acc)

	    torch.save( best_model.state_dict(), "./Models/model_{date}_{acc}".format(date=str(np.datetime64('now')), acc=acc) )


if __name__ == '__main__':
    main()

