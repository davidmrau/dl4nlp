# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch.utils.data as data
from collections import Counter
from sklearn.utils import shuffle

class LanguageDataset(data.Dataset):

    def __init__(self, filename, seq_length, ref=None):
        data = open(filename, 'r', encoding='utf-8').read().split('\n')
        self._inputs = []
        self._targets = []
        count = 0
        for line in data:
            if line != '':
                line_s = line.split('\t')
                if len(list(line_s[1])) >= seq_length:
                    self._inputs.append(line_s[1])
                    self._targets.append(line_s[0])
                else:
                    count += 1

        print('{} paragraphs are too short.'.format(count))
        self._inputs, self._targets  = shuffle(self._inputs, self._targets)
        self._offset = 0
        if ref is None:
            self._langs = list(set(self._targets))
            self._seq_length = seq_length
            self._chars = list(set(''.join(self._inputs)))
            self._data_size, self._vocab_size, self._n_langs = len(self._targets), len(self._chars), len(self._langs)
            print("Initialize dataset with {} characters {} langs.".format(
                self._vocab_size, self._n_langs))
            self._char_to_ix = { ch:i for i,ch in enumerate(self._chars) }
            self._ix_to_char = { i:ch for i,ch in enumerate(self._chars) }

            self._lang_to_ix = { l:i for i,l in enumerate(self._langs) }
            self._ix_to_lang = { i:l for i,l in enumerate(self._langs) }
        else:
            self._langs = ref._langs
            self._seq_length = ref._seq_length
            self._chars = ref._chars
            self._data_size, self._vocab_size, self._n_langs = len(self._targets), len(self._chars), len(self._langs)
            print("Initialize dataset with {} characters {} langs.".format(
                self._vocab_size, self._n_langs))
            self._char_to_ix = ref._char_to_ix
            self._ix_to_char = ref._ix_to_char
            self._lang_to_ix = ref._lang_to_ix
            self._ix_to_lang = ref._ix_to_lang
    def __getitem__(self, index):
        char_list = list(self._inputs[index])
        #if len(char_list)-self._seq_length-1 <= 0:
        #    print('sdfasdfasdf', len(char_list),self._seq_length-13 )
        if len(char_list) - self._seq_length-1 <= 0:
            offset = 0
        else:
            offset = np.random.randint(0, len(char_list)-self._seq_length-1)
        inputs = np.zeros((len(list(self._inputs[index])[offset:offset+self._seq_length]), self._vocab_size))
        for i,ch in enumerate(list(char_list)[offset:offset+self._seq_length]):
            inputs[i, self._char_to_ix[ch]] = 1

        #targets[np.array(self._lang_to_ix[self._targets[index]])]=1
        targets = np.array(self._lang_to_ix[self._targets[index]])

        #inputs =  np.array([self._char_to_ix[ch] for ch in char_list])[offset:offset+self._seq_length]
        return inputs, targets
    # a simple custom collate function, just to show the idea

    def convert_to_string(self, char_ix):
        return ''.join(self._ix_to_char[ix] for ix in char_ix)

    def __len__(self):
        return self._data_size

    @property
    def vocab_size(self):
        return self._vocab_size


    @property
    def n_langs(self):
        return self._n_langs
