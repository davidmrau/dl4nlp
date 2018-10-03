import os
import time
from datetime import datetime
import argparse
import random
import numpy as np
import pickle

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LanguageDataset
from model import GRU
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
################################################################################
from sklearn.metrics import classification_report


def train(config):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    dtype = torch.cuda.LongTensor if use_cuda else torch.LongTensor
    # Initialize the device which to run the model on
    device = torch.device(device)
    # Initialize the dataset and data loader (note the +1)
    # Initialize the dataset and data loader (note the +1)
    train_dataset = pickle.load(open(config.dataset+'/train_dataset.p', 'rb'))
    test_dataset = pickle.load(open(config.dataset+'/test_dataset.p', 'rb'))
    data_loader_train = DataLoader(train_dataset, config.batch_size, num_workers=1)
    data_loader_test = DataLoader(test_dataset, config.batch_size, num_workers=1)
    # Initialize the model that we are going to use
    labels = [train_dataset._ix_to_lang[l] for l in range(train_dataset.n_langs)]
    model = GRU(config.batch_size, train_dataset.vocab_size, train_dataset.n_langs, \
     config.gru_num_hidden, config.gru_num_layers, config.dropout_keep_prob).to(device)
    model.load_state_dict(torch.load(config.model))
    model.eval()
    predictions = []
    targets = []
    for step, (batch_inputs, batch_targets) in enumerate(data_loader_test):
        if step % 100 is 0:
            print('step {}'.format(step))
        y_pred_batch, _ = model(batch_inputs.type(dtype).float())
        y_pred_batch = y_pred_batch.transpose(1,0)[-1]
        y_pred_batch = y_pred_batch.argmax(1)
        for pred in y_pred_batch.flatten().cpu().detach().numpy():
            predictions.append(pred)
        for tar in batch_targets.flatten().cpu().detach().numpy():
            targets.append(tar)

    print(classification_report(targets, predictions, target_names=labels, labels=np.arange(len(labels))))

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Path to model") 
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset files")
    
    parser.add_argument('--gru_num_hidden', type=int, default=128, help='Number of hidden units in the GRU')
    parser.add_argument('--gru_num_layers', type=int, default=2, help='Number of GRU layers in the model')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.8, help='Dropout keep probability')
    # Training params

    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    config = parser.parse_args()

    # Train the model
    train(config)
