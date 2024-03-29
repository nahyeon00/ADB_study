from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, BertTokenizer, AdamW, BertConfig
import pandas as pd
import torch.nn as nn
import numpy as np
import torch
import argparse
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
import random
from transformers import AdamW, get_linear_schedule_with_warmup

from data_info import * 

class ADBDataset(Dataset):
    def __init__(self, process_data, max_seq_len, label_list):
        self.data = process_data

        self.max_seq_len = max_seq_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True) 
        self.label_list = label_list
        self.label_map = {}
        for i, label in enumerate(self.label_list):
            self.label_map[label] = i
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        data = self.data[index]
        doc = data['text']

        features = self.tokenizer(str(doc), padding='max_length', max_length= self.max_seq_len, truncation=True, return_tensors='pt') 

        input_ids = features['input_ids'].squeeze(0)
        attention_mask = features['attention_mask'].squeeze(0)
        token_type_ids = features['token_type_ids'].squeeze(0)
        ori_label = data['label']
        label_id = torch.tensor([self.label_map[ori_label]], dtype = torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'label_id': label_id
        }



class ADBDataModule(pl.LightningDataModule):
    def __init__(self, args):
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.train_data_path = f'{self.data_path}/train.tsv'
        self.val_data_path = f'{self.data_path}/dev.tsv'
        self.test_data_path = f'{self.data_path}/test.tsv'

        self.max_seq_len = max_seq_lengths[self.dataset]
        self.batch_size = args.batch_size
        self.known_cls_ratio = args.known_cls_ratio
        self.worker = args.num_workers
        self.labeled_ratio = args.labeled_ratio
        self.mode = args.mode

        ####### label list
        self.all_label_list = benchmark_labels[self.dataset]
        self.n_known_cls = round(len(self.all_label_list) * self.known_cls_ratio)
        self.known_label_list = np.random.choice(np.array(self.all_label_list, dtype=str), self.n_known_cls, replace=False)
        self.known_label_list = self.known_label_list.tolist()
        print("known_label_list", self.known_label_list)

        args.num_labels = self.num_labels = len(self.known_label_list)
        print("num_labels", self.num_labels)


        if self.dataset == 'oos':
            self.unseen_label = 'oos'
        else:
            self.unseen_label = '<UNK>'
        
        self.unseen_label_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_label]
        
    def setup(self, stage):

        if stage in (None, 'fit'):
            # prepare
            self.train_data = pd.read_csv(self.train_data_path, delimiter="\t")
            self.valid_data = pd.read_csv(self.val_data_path, delimiter="\t")

            self.train_examples = []
            self.val_examples = []

            for i in self.train_data.index:
                cur_label = self.train_data.loc[i]['label']
                if (cur_label in self.known_label_list) and (np.random.uniform(0, 1) <= self.labeled_ratio):
                    self.train_examples.append(self.train_data.iloc[i])
            for i in self.valid_data.index:
                cur_label = self.valid_data.loc[i]['label']
                if (cur_label in self.known_label_list):
                    self.val_examples.append(self.valid_data.iloc[i])

            self.train = ADBDataset(self.train_examples, self.max_seq_len, self.label_list)
            self.valid = ADBDataset(self.val_examples, self.max_seq_len, self.label_list)

        elif stage in (None, 'test'):
            # prepare
            self.test_data = pd.read_csv(self.test_data_path, delimiter="\t")

            self.test_examples = []
            if self.mode == 'feature_train':
                for i in self.test_data.index:
                    cur_label = self.test_data.loc[i]['label']
                    if (cur_label in self.known_label_list):
                        self.test_examples.append(self.test_data.iloc[i])
            else:
                for i in self.test_data.index:
                    cur_label = self.test_data.loc[i]['label']
                    if (cur_label in self.label_list) and (cur_label is not self.unseen_label):
                        self.test_examples.append(self.test_data.iloc[i])
                    else:
                        self.test_data.loc[i]['label'] = self.unseen_label
                        self.test_examples.append(self.test_data.iloc[i])

            self.test = ADBDataset(self.test_examples, self.max_seq_len, self.label_list)
        
    def train_dataloader(self):
        sampler = RandomSampler(self.train)
        return DataLoader(self.train, batch_size=self.batch_size, num_workers= self.worker, sampler = sampler)
    
    def val_dataloader(self):
        sampler = SequentialSampler(self.valid)
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers= self.worker, sampler = sampler)
    
    def test_dataloader(self):
        sampler = SequentialSampler(self.test)
        return DataLoader(self.test, batch_size=self.batch_size, num_workers= self.worker, sampler = sampler)
    
    def predict_dataloader(self):
        sampler = RandomSampler(self.train)
        return DataLoader(self.train, batch_size=self.batch_size, num_workers= self.worker, sampler = sampler)
