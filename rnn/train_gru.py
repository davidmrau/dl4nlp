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
    train_dataset = LanguageDataset(config.train)
    test_dataset = LanguageDataset(config.test, train_dataset)
    data_loader_train = DataLoader(train_dataset, config.batch_size, num_workers=1)
    data_loader_test = DataLoader(test_dataset, config.batch_size, num_workers=1)
    pickle.dump(train_dataset, open(config.save_path+'/train_dataset.p', 'wb'))
    pickle.dump(test_dataset, open(config.save_path+'/test_dataset.p', 'wb'))
   # Initialize the model that we are going to use
    model = GRU(config.batch_size, train_dataset.vocab_size, train_dataset.n_langs, \
     config.gru_num_hidden, config.gru_num_layers, config.dropout_keep_prob).to(device)
    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), config.learning_rate)
    epochs = 10
    lr = config.learning_rate
    for epoch in range(epochs):
        for step, (batch_inputs, batch_targets) in enumerate(data_loader_train):

            model.train()
            # Only for time measurement of step through network
            t1 = time.time()
            optimizer.zero_grad()
            # batch_inputs.sort(key=len, reverse=True)
            # emb = nn.Embedding(train_dataset.vocab_size, 64).cuda()
            # packed = rnn_utils.pack_sequence([torch.Tensor(item).type(dtype) for item in batch_inputs])
            # embedded = rnn_utils.pack_sequence(emb(packed.data).unsqueeze(0))
            # initialize one hot
            y_pred_batch, _ = model(batch_inputs.type(dtype).float())
            y_pred_batch = y_pred_batch.transpose(1,0)[-1]
            loss = criterion(y_pred_batch, batch_targets.to(device))
            #torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=config.max_norm)
            loss.backward()
            optimizer.step()

            loss = loss.item()
            accuracy = np.sum(np.argmax(y_pred_batch.cpu().detach().numpy(), axis=1) == batch_targets.cpu().detach().numpy())/batch_targets.shape[0]
            # Just for time measurement
            t2 = time.time()
            examples_per_second = config.batch_size/float(t2-t1)
            config.train_steps = int(config.train_steps)

            if step == config.train_steps:
                # If you receive a PyTorch data-loader error, check this bug report:
                # https://github.com/pytorch/pytorch/pull/9655
                break
            # save model
            if step % config.learning_rate_step is 0 and step is not 0:
                lr = lr*config.learning_rate_decay
                print('Learning rate decreased: {} \n'.format(lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            if step % config.save_every == 0:
                print('Save model')
                torch.save(model.state_dict(), config.save_path+'epoch_{}_step_{}.pth'.format(epoch, step))
            if step % config.print_every == 0:

                print("Epoch {} [{}] Train Step {:04d}/{:04d}, Batch Size = {}, Examples/Sec = {:.2f}, "
                     "Accuracy = {:.2f}, Loss = {:.3f}".format(epoch,
                       datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                       config.train_steps, config.batch_size, examples_per_second,
                       accuracy, loss
                ))
            if step % config.evaluate_every ==0:
                model.eval()
                accuracies = []
                for step, (batch_inputs, batch_targets) in enumerate(data_loader_test):
                    if step == 20:
                        break
                    y_pred_batch, _ = model(batch_inputs.type(dtype).float())
                    y_pred_batch = y_pred_batch.transpose(1,0)[-1]
                    accuracy = np.sum(np.argmax(y_pred_batch.cpu().detach().numpy(), axis=1) == batch_targets.cpu().detach().numpy())/batch_targets.shape[0]
                    accuracies.append(accuracy)
                print('Average accuracy: {}'.format(np.mean(accuracies)))
    model.eval()
    accuracies = []
    for batch_inputs, batch_targets in data_loader_test:
        y_pred_batch, _ = model(batch_inputs.type(dtype).float())
        y_pred_batch = y_pred_batch.transpose(1,0)[-1]
        accuracy = np.sum(np.argmax(y_pred_batch.cpu().detach().numpy(), axis=1) == batch_targets.cpu().detach().numpy())/batch_targets.shape[0]
        accuracies.append(accuracy)

    print('Final accuracy on the test set: {}'.format(np.mean(accuracies)))
    torch.save(model.state_dict(), config.save_path+'_last_model.pth'.format(step))
    print('Done training.')


 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--test', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--train', type=str, required=True, help="Path to a .txt file to test on")
    parser.add_argument('--gru_num_hidden', type=int, default=128, help='Number of hidden units in the GRU')
    parser.add_argument('--gru_num_layers', type=int, default=2, help='Number of GRU layers in the model')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.8, help='Dropout keep probability')
    # Training params
    parser.add_argument('--learning_rate_decay', type=float, default=0.2, help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000, help='Learning rate step')

    parser.add_argument('--batch_size', type=int, default=64, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=1e6, help='Number of training steps')
    parser.add_argument('--save_path', type=str, default="./models/", help='Output path for models and dataset files')
    parser.add_argument('--print_every', type=int, default=10, help='How often to print training progress')
    parser.add_argument('--evaluate_every', type=int, default=500, help='How often the model is evaluated')
    parser.add_argument('--save_every', type=int, default=500, help='How often to save the model')
    config = parser.parse_args()

    # Train the model
    train(config)
