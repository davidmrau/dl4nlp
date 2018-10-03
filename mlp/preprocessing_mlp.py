"""# Preprocessing MLP"""

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
import os
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer

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
      # input:
      #  path_x: (str) path of x
      #  path_y: (str) path of y
      # return:
      #   dict: {language: [paragraph, ... ,paragraph], language: [...}
      with open(path_x, 'r') as f:
        paragraphs = f.read().split('\n')

      with open(path_y, 'r') as f:
        languages = f.read().split('\n')

      data = defaultdict(lambda: [])
      for lang, para in zip(languages, paragraphs):
        para.replace('.', '')
        para = re.sub(' +', '',para)
        data[lang].append(list(para))
      return data


    def read_files_n(path_x, path_y, n):
      # input:
      #  path_x: (str) path of x
      #  path_y: (str) path of y
      # return:
      #   dict: {language: [paragraph, ... ,paragraph], language: [...}
      with open(path_x, 'r') as f:
        paragraphs = f.read().split('\n')

      with open(path_y, 'r') as f:
        languages = f.read().split('\n')

      data = defaultdict(lambda: [])
      count = 0
      for lang, para in zip(languages, paragraphs):
        count+= 1
        if count == n:
          break
        para.replace('.', '')
        para = re.sub(' +', '',para)
        data[lang].append(list(para))

      return data


    def preprocess(train_dict,test_dict):
      # input:
      #   lang_dict: (dict) {language: [paragraph, ... ,paragraph], language: [...}
      # return:
      language = Lang()
      train_data = []
      y_train = []
      [language.addLang(key) for key in train_dict.keys()]
      for lang, paragraphs in train_dict.items():
        for para in paragraphs:
          train_data.append(''.join(para))
          y_train.append(language.one_hot(lang))
      vec = TfidfVectorizer(min_df=config.min_char_freq, analyzer='char', norm='l2')
      vec.fit(train_data)
      X_train = vec.transform(train_data)

      del train_data
      test_data = []
      y_test = []
      for lang, paragraphs in test_dict.items():
        for para in paragraphs:
          test_data.append(''.join(para))
          y_test.append(language.one_hot(lang))

      X_test = vec.transform(test_data)

      return X_train, np.array(y_train), X_test, np.array(y_test), language.index2lang


    # read files
    train_dict = read_files(config.X_train, config.y_train)
    test_dict = read_files(config.X_test, config.y_train)
    # pre-process
    X_train, y_train, X_test, y_test, index2lang = preprocess(train_dict, test_dict)
    # create folder if it doesn't exist
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    # save files
    pickle.dump([X_train, y_train], open(config.save_path+'/train.p', 'wb'))
    pickle.dump([X_test, y_test], open(config.save_path+'/test.p', 'wb'))
    pickle.dump(index2lang, open(config.save_path+'/index2lang.p', 'wb'))


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
    parser.add_argument('--save_path', type=str, default="./data/", help='Output path for preprocessed training data: "train.p",\
     preprocessed test data:"test.p" and the index2lang dicttionary: "index2lang.p"')
    config = parser.parse_args()

    # preprocess
    process(config)
