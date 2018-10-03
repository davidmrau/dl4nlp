"""## Visualization"""

import argparse
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
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import matplotlib as mpl

def read_file(path_x):
  with open(path_x, 'r') as f:
    paragraphs = f.read()
    all_chars = list(set(paragraphs))
    paragraphs= paragraphs.split('\n')

  data = defaultdict(lambda: [])
  for para in paragraphs:
    spl = para.split('\t')
    if len(spl) == 2:
        lang,para = spl[0], spl[1]
        data[lang].append(para)
  return data

def vis(config):
    # load files
    train_dict = read_file(config.train)
    legal_chars = pickle.load(open(config.legal_chars, 'rb'))
    accuracies_dict = pickle.load(open(config.acc_dict, 'rb'))


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
        colors.append(cmap(float(accuracies_dict[lang])))

    # plot
    fig = plt.figure(figsize=(12, 10))
    for i,c,label in zip(target_ids, colors, target_names):
      plt.scatter(X_2d[i][1],X_2d[i][0],c=c, label=label)
      acc = float(accuracies_dict[label])
      if acc < config.label_threshold:
          plt.annotate(label,xy=(X_2d[i][1],X_2d[i][0]))
      # if 'zho' in label or 'zh-yue' in label or 'wuu' in label:
      #     plt.annotate(label,xy=(X_2d[i][1],X_2d[i][0]))
    plt.savefig(config.save_fig)




 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--train', type=str, required=True, help="Path to the preprocessed test data")
    parser.add_argument('--legal_chars', type=str, required=True, help="Path legal chars")
    parser.add_argument('--acc_dict', type=str, required=True, help="Path to accuracy dict")
    parser.add_argument('--save_fig', type=str, required=True, help="Path for saving figure")
    parser.add_argument('--label_threshold', type=float, default:0.8, help="acc threshold for showing labels")

    config = parser.parse_args()

    # Train the model
    vis(config)
