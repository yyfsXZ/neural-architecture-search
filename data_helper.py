#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_helper.py
# @Author: zhangxiang
# @Date  : 2019/7/16
# @Desc  :

import sys
import os
import random

sys.path.append(os.path.join(os.path.dirname(__file__),".."))
sys.path.append(os.path.join(os.path.dirname(__file__),"../.."))
sys.path.append(os.path.join(os.path.dirname(__file__),"../../.."))

from CommonLibs.DataHelper import CharHelper
from CommonLibs.FileIoUtils import MySentences

class MyHelper():
    def __init__(self, max_seq_len):
        self.char_helper = CharHelper()
        self.label2id = {}
        self.id2lable = {}
        self.label_num = 0
        self.max_seq_len = max_seq_len

    def initialize(self):
        self.char_helper.initialize()

    def read_input(self, filename):
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for line in MySentences(filename):
            splits = line.strip("\r\n").replace(" ", "").lower().split('\t')
            if len(splits) != 2:
                continue
            query = splits[0]
            label = splits[1]
            if label not in self.label2id:
                self.label2id[label] = self.label_num
                self.id2lable[str(self.label_num)] = label
                self.label_num += 1
            query_ids = self.char_helper.trans_query_to_ids(query)
            if len(query_ids) < self.max_seq_len:
                query_ids.extend([self.char_helper.get_padd_char_id()] * (self.max_seq_len - len(query_ids)))
            else:
                query_ids = query_ids[:self.max_seq_len]
#            print(query_ids)
            label_id = self.label2id[label]

            if random.uniform(0, 1) <= 0.8:
                train_x.append(query_ids)
                train_y.append(label_id)
            else:
                test_x.append(query_ids)
                test_y.append(label_id)
        return train_x, train_y, test_x, test_y

    def get_vocab_size(self):
        return self.char_helper.get_vocab_size()

    def get_label_size(self):
        return len(self.label2id.keys())

    def get_char_by_id(self, id_):
        return self.char_helper.get_char_by_id(id_)
