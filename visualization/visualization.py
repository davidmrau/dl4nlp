"""## Visualization"""


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
from google.colab import files
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib as mpl

# load files

train_dict = pickle.load(open(train_rnn.txt, 'rb'))
legal_chars = pickle.load(open(legal_chars.txt, 'rb'))
accuracies_dict = pickle.load(open(accuracies.txt, 'rb'))


# initit dataframe
train_df = pd.DataFrame.from_dict(train_dict)

# create word freq vector for every language
lang_vectors = []
count_dict = { c:0 for c in legal_chars}
for lang in train_df.columns:
  joined_paragraphs = ''.join(train_df[lang].values)
  for char, count in Counter(joined_paragraphs).items():
    count_dict[char] = count

  lang_vectors.append(list(count_dict.values()))


# use T-SNE to project into 2D
tsne = TSNE(n_components=2, random_state=0)
X_2d = tsne.fit_transform(lang_vectors)
target_names = train_df.columns
target_ids = range(len(train_df.columns))
cmap = mpl.cm.plasma
# map acuracies to colors
colors = []
for lang in target_names:
  colors.append(cmap(accuracies_dict[lang]))

# plot
fig = plt.figure(figsize=(12, 10))
for i,c,label in zip(target_ids, colors, target_names):
  plt.scatter(X_2d[i][1],X_2d[i][0],c=c,  label=label)
plt.show()
