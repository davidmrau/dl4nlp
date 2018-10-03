import torch
from sklearn import metrics
import pickle
from mlp import MLP
import numpy as np
import argparse

def eval(config):
    X_test, y_test = pickle.load(open(config.test, 'rb'))
    X_test = X_test.todense()
    index2lang = pickle.load(open(config.index2lang, 'rb'))
    labels = [index2lang[i] for i in range(len(index2lang))]
    model = MLP(X_test.shape[1],[config.num_hidden],y_test.shape[1])
    model.load_state_dict(torch.load(config.model))
    model.eval()
    y_pred = model(torch.from_numpy(X_test).type(torch.FloatTensor))
    y_pred = np.argmax(y_pred.cpu().detach().numpy(), axis=1)
    t = np.argmax(y_test, axis=1)
    t = t.reshape(1,-1).squeeze()
    print(metrics.classification_report(t, y_pred, target_names=labels, labels=np.arange(len(index2lang))))
#print(metrics.precision_recall_fscore_support(t, y_pred))

 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--test', type=str, required=True, help="Path to the preprocessed test data")
    parser.add_argument('--index2lang', type=str, required=True, help="Path index2lang dict")
    parser.add_argument('--model', type=str, required=True, help="Path to model")
    parser.add_argument('--num_hidden', type=int, default=512, help='Number of hidden units')
    config = parser.parse_args()


    # evaluate the model
    eval(config)
