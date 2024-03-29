from torch.utils.data import Dataset, DataLoader
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
from data_preprocess import *


class BERTfeature(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)

        # data
        self.dataset = args.dataset  # 저장 파일명 위해 필요
        self.known_cls_ratio = args.known_cls_ratio  # 저장 파일명 위해 필요

        self.num_labels = args.num_labels

        # use pretrained BERT
        model_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bert = BertModel(model_config)
        self.dense = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout =  nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

        self.__build_loss()

        # centroids 초기화
        self.centroids = torch.zeros(self.num_labels, 768)
        self.total_labels = torch.empty(0, dtype=torch.long)
        
        # # weight initialization
        # self.init_weights()
        # self.weight = Parameter(torch.FloatTensor(args.num_labels, args.feat_dim).to(args.device))
        # nn.init.xavier_uniform_(self.weight)


    def forward(self, input_ids, attention_mask, token_type_ids):
        # output [last_hidden_state, pooler_output, hidden_states]  -> last hidden layer
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # [128, 45, 768]
        last_hidden_layer = outputs[0]

        # mean pooling [128, 768]
        mean_pooling = last_hidden_layer.mean(dim=1)
        
        # dense layer [128, 768]
        pooled_output = self.dense(mean_pooling)
        
        # activation function & dropout
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)

        # classifier [128, num_label]
        logits = self.classifier(pooled_output)

        return pooled_output, logits
        

    
    def training_step(self, batch, batch_idx):
        print("training step")
        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label_id = batch['label_id']
        
        # fwd
        _, logits = self.forward(input_ids, attention_mask, token_type_ids)

        # loss
        print("lo", logits.size(), label_id.size(), label_id.squeeze(-1).size())

        loss = self._loss(logits, label_id.long().squeeze(-1))
        
        # logs
        tensorboard_logs = {'train_loss': loss}

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        print("start validation")
        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label_id = batch['label_id']
         
        # fwd
        _, logits = self.forward(input_ids, attention_mask, token_type_ids)

        total_probs = F.softmax(logits.detach(), dim=1)

        total_maxprobs, total_preds = total_probs.max(dim = 1)
   
        y_pred = total_preds.cpu().numpy()
        y_true = label_id.cpu().numpy()

        val_acc = accuracy_score(y_true, y_pred)
        eval_score = round(val_acc * 100, 2)

        self.log('val_acc', val_acc)

        return val_acc
    
    
    
    def test_step(self, batch, batch_nb):
        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label_id = batch['label_id']
        
        _, logits = self.forward(input_ids, attention_mask, token_type_ids)

        total_probs = F.softmax(logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_pred = total_preds.cpu().numpy()
        y_true = label_id.cpu().numpy()
        test_acc = accuracy_score(y_true, y_pred)
        eval_score = round(test_acc * 100, 2)
        test_acc = torch.tensor(test_acc)

        
        self.log_dict({'test_acc': test_acc})
        
        return {'test_acc': test_acc}
    
    
    def predict_step(self, batch, batch_idx):
        print("predict step")

        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label_ids = batch['label_id']

        # fwd
        pooled_output, _ = self.forward(input_ids, attention_mask, token_type_ids)

        self.total_labels = torch.cat((self.total_labels.to(self.device), label_ids))

        self.centroids = self.centroids.to(self.device)
        print("centrodi", self.centroids.dtype)
        # assert 1==0
        print("len", len(label_ids))

        for i in range(len(label_ids)):
            label = label_ids[i]
            self.centroids[label] += pooled_output[i]
            print("feature", pooled_output[i])

        return {'centroids':self.centroids, 'total_labels':self.total_labels}

    
    def on_predict_end(self):
        print("end predict centroids", self.centroids)

        self.total_labels = self.total_labels.cpu().numpy()
        print("total label", self.class_count(self.total_labels))
        self.centroids /= torch.tensor(self.class_count(self.total_labels)).float().unsqueeze(1).to(self.device)

        print('finish cal centroids', self.centroids)
        path = '/workspace/intent/newADB/centroids/'
        file_name = f'centroids_{self.dataset}_{self.known_cls_ratio}.npy'
        np.save(os.path.join(path, file_name), self.centroids.detach().cpu().numpy())
    
    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num
    
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        parameters = []
        for p in self.parameters():
            if p.requires_grad:
                parameters.append(p)
            else:
                print(p)
                
        optimizer = torch.optim.Adam(parameters, lr=2e-05 , eps=1e-08)

        return optimizer
    
    def __build_loss(self):
        self._loss = nn.CrossEntropyLoss()

