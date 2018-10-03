"""# Preprocessing GRU"""

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
import argparse


def process(config):
    class Lang:
        def __init__(self):
            self.lang2index = {}
            self.index2lang = {}
            self.n_langs = 0

        def addLang(self, lang):
            if lang not in self.lang2index:
              self.lang2index[lang] = self.n_langs
              self.index2lang[self.n_langs] = lang
              self.n_langs += 1

        def one_hot(self, lang):
          one_hot_encoded = np.zeros(self.n_langs)
          one_hot_encoded[self.lang2index[lang]] = 1
          return one_hot_encoded

    def read_files(path_x, path_y):
      with open(path_x, 'r') as f:

        paragraphs = f.read()
        paragraphs = paragraphs.replace('.', '')
        paragraphs = re.sub(' +', '',paragraphs)
        all_chars = list(set(paragraphs))
        paragraphs= paragraphs.split('\n')

      with open(path_y, 'r') as f:
        languages = f.read().split('\n')

      data = defaultdict(lambda: [])
      for lang, para in zip(languages, paragraphs):
        if lang != '':
          data[lang].append(para)
      return data, all_chars

    def get_legal_chars(dict_, all_chars):
      legal_chars = []
      for lang, paragraphs in dict_.items():
        joined_chars = ''.join(paragraphs)
        c = Counter(joined_chars)
        df = pd.DataFrame.from_dict(c, orient='index')
        filtered = df[df >= config.min_char_freq].dropna().index
        for char in list(filtered):
          legal_chars.append(char)
      return ''.join(list(set(legal_chars)))


    def filter_chars(dict_, legal_chars):
      res = {}
      for lang, paragraphs in dict_.items():
        lang_paras = []
        for para in dict_[lang]:
          lang_paras.append(''.join([char  for char in para if char in legal_chars]))
        res[lang] = lang_paras
      return res

    def write_to_file(data, filename):
      with open(filename, 'a', encoding='utf-8') as file:
        for lang, pargraphs in data.items():
          for para in pargraphs:
            file.write('{}\t{}\n'.format(lang, para))

    def chars_to_index(chars):
      return { ch:i for i,ch in enumerate(chars)}

    # read files
    train_dict, all_chars = read_files(config.X_train, config.y_train)
    test_dict,_ = read_files(config.X_test, config.y_test)
    # get chars with total corpus freq > min_char_freq
    legal_chars = get_legal_chars(train_dict, all_chars)
    # build char2index dict
    chars_to_index = chars_to_index(legal_chars)

    print('Number of legal chars', len(legal_chars))
    # filter data
    train_dict_filtered = filter_chars(train_dict, legal_chars)
    test_dict_filtered = filter_chars(test_dict, legal_chars)
    # write to file

    write_to_file(train_dict_filtered, config.save_path+'/train.txt' )
    write_to_file(test_dict_filtered, config.save_path+'/test.txt' )
    pickle.dump(index2lang, open(config.save_path+'/index2lang_rnn.p', 'wb'))
    pickle.dump(legal_chars, open(config.save_path+'/legal_chars.p', 'wb'))





 ################################################################################
 ################################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--X_train', type=str, required=True, help="Path to the training input data")
    parser.add_argument('--y_train', type=str, required=True, help="Path to the preprocessed target train data")
    parser.add_argument('--X_test', type=str, required=True, help="Path to the test input data")
    parser.add_argument('--y_test', type=str, required=True, help="Path to the target test data")
    parser.add_argument('--min_char_freq', type=int, default=100, help="use only chars with frequency >= min_char_freq")
    parser.add_argument('--save_path', type=str, default="./data/", help='Output path for preprocessed training data: "train.txt",\
     preprocessed test data:"test.txt", legal_chars: "legal_chars.p" and the index2lang dicttionary: "index2lang.p"')
    config = parser.parse_args()

    # preprocess
    process(config)
