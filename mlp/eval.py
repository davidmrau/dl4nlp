import torch
from sklearn import metrics
import pickle
from mlp import MLP
import numpy as np
X_test, y_test = pickle.load(open('test.p', 'rb'))
X_test = X_test.todense()
index2lang = pickle.load(open('index2lang.p', 'rb'))
labels = [index2lang[i] for i in range(len(index2lang))]
model = MLP(X_test.shape[1],[512],y_test.shape[1])
model.load_state_dict(torch.load('model.pth'))
model.eval()
print(labels)
y_pred = model(torch.from_numpy(X_test).type(torch.FloatTensor))
y_pred = np.argmax(y_pred.cpu().detach().numpy(), axis=1)
t = np.argmax(y_test, axis=1)
t = t.reshape(1,-1).squeeze()
print(y_pred.shape, t.shape)
print(t)
print(y_pred)
print(metrics.classification_report(t, y_pred, target_names=labels, labels=np.arange(len(index2lang))))
#print(metrics.precision_recall_fscore_support(t, y_pred))

