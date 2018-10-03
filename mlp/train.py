import os
from collections import Counter, defaultdict
import string
import numpy as np
import re
import pandas as pd
import torch
import torch.nn as nn
import random
import math
import pickle
from sklearn import metrics
from mlp import MLP
import argparse

"""#Training"""
#taken from
# https://medium.com/python-learning-notes-those-cool-stuff/mini-batch-gradient-descent-ac015a8e4acc
def random_mini_batches(X, Y, mini_batch_size, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[0]                  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:]
    shuffled_Y = Y[permutation,:].reshape((m,-1))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k*mini_batch_size : (k+1)*mini_batch_size,:]
        mini_batch_Y = shuffled_Y[k*mini_batch_size : (k+1)*mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches*mini_batch_size : m,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches*mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches



def accuracy(predictions, targets):
  accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(targets, axis=1))/targets.shape[0]
  return accuracy



def train(config):
    X_train, y_train = pickle.load(open(config.train, 'rb'))
    X_test, y_test = pickle.load(open(config.test, 'rb'))


    # sparse to dense
    X_train = X_train.todense()
    X_test = X_test.todense()

    minibatch_size = config.batch_size
    epochs = config.epochs

    # Device configuration
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Running in GPU model')

    device = torch.device('cuda' if use_cuda else 'cpu')
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor


    model = MLP(X_train.shape[1], [config.num_hidden], y_train.shape[1]).to(device)

    # intialize loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    accuracies = []
    eval_losses = []
    losses = []
    for e in range(epochs):
      total_loss = 0
      batches = random_mini_batches(X_train, y_train, minibatch_size)
      for X_train_b, y_train_b in batches:
          model.train()
          # Forward pass
          y_pred = model(torch.from_numpy(X_train_b).type(dtype))
          y_train_b = torch.from_numpy(y_train_b).type(dtype)
          t = torch.max(y_train_b,1)[1]
          loss = criterion(y_pred, t)
          total_loss += loss.item()
          # Backward and optimize
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      # evaluate
      model.eval()
      # Forward pass
      batches = random_mini_batches(X_test, y_test, minibatch_size)
      total_eval_batch_loss = 0
      total_acc = 0
      for X_test_b, y_test_b in batches:
        # Forward pass
        y_pred = model(torch.from_numpy(X_test_b).type(dtype))
        y_test_b = torch.from_numpy(y_test_b).type(dtype)
        t = torch.max(y_test_b,1)[1]
        loss = criterion(y_pred, t)
        total_eval_batch_loss += loss.item()
        acc = metrics.accuracy_score(np.argmax(y_pred.cpu().detach().numpy(),axis=1),np.argmax(y_test_b.cpu().detach().numpy(), axis=1))
        total_acc += acc
      av_acc = total_acc/len(batches)
      av_eval_batch_loss = total_eval_batch_loss/len(batches)
      av_loss = total_loss/len(batches)
      accuracies.append(av_acc)
      eval_losses.append(av_eval_batch_loss)
      losses.append(av_loss)
      print('epoch: {}'.format(e))
      print(minibatch_size)
      print('accuracy: {}'.format(av_acc))
      print('av val batch loss: {}'.format(av_eval_batch_loss))
      print('av batch loss: {}'.format(av_loss))
      print()
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    pickle.dump(accuracies, open(config.save_path+'/acc.p', 'wb'))
    pickle.dump(eval_losses, open(config.save_path+'/eval_loss.p', 'wb'))
    pickle.dump(losses, open(config.save_path+'/loss.p', 'wb'))
    torch.save(model.state_dict(), config.save_path+'/model.pth')



 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--test', type=str, required=True, help="Path to the preprocessed test data")
    parser.add_argument('--train', type=str, required=True, help="Path to the preprocessed train data")
    parser.add_argument('--num_hidden', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of examples to process in a batch')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--save_path', type=str, default="./models/", help='Output path for accuracy: "acc.p",\
     evaluation loss:"eval_loss.p", loss: "loss.p" and the model: "model.pth"')
    config = parser.parse_args()

    # Train the model
    train(config)
