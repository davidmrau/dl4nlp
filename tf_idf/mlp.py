import torch
from torch import nn

class MLP(nn.Module):
  """
  This class implements a Multi-layer Perceptron in PyTorch.
  It handles the different layers and parameters of the model.
  Once initialized an MLP object can perform forward.
  """

  def __init__(self, n_inputs, n_hidden, n_classes):
    super(MLP, self).__init__()
    self.modules = []
    # if list is empty -> no linear layer
    prev_size = n_inputs
    if len(n_hidden) > 0:
        for n in range(len(n_hidden)):
            self.modules.append(nn.Linear(prev_size, n_hidden[n]))
            prev_size = n_hidden[n]
    self.modules.append(nn.Linear(prev_size, n_classes))

    self.relu = nn.ReLU()
    self.layers = nn.ModuleList(self.modules)

  def forward(self, x):
    out = x
    for i,m in enumerate(self.layers):
        # if not last layer
        if i == len(self.layers)-1:
            out = m(out)
        elif i != len(self.layers):
            out = m(out)
            out = self.relu(out)
    return out
