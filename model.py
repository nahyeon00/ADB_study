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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# class pre(BertPreTrainedModel):
#     def __init__(self, args):


class BERTfeature(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        set_seed(0)

        # data
        self.dataset = args.dataset
        self.known_cls_ratio = args.known_cls_ratio

        self.all_label = benchmark_labels[args.dataset]
        self.n_known_cls = round(len(self.all_label)*self.known_cls_ratio)

        self.known_label_list = np.random.choice(np.array(self.all_label, dtype=str), self.n_known_cls, replace=False)
        self.known_label_list = self.known_label_list.tolist()

        self.num_labels = len(self.known_label_list)

        # use pretrained BERT
        model_config = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=model_config, )
        self.dense = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.activation = nn.ReLU()
        self.dropout =  nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

        self.__build_loss()

        # centroids 초기화
        self.centroids = torch.zeros(self.num_labels, 768).to(self.device)

        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        self.total_logits = torch.empty((0, self.num_labels)).to(self.device)
        
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
        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label_id = batch['label_id']
        
        # fwd
        _, logits = self.forward(input_ids, attention_mask, token_type_ids)
        
        # loss
        loss = self._loss(logits, label_id.long().squeeze(-1))
        
        # logs
        tensorboard_logs = {'train_loss': loss}

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    # def on_validation_start(self):
    #     self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
    #     self.total_logits = torch.empty((0, self.num_labels)).to(self.device)

    #     return self.total_labels, self.total_logits

    def validation_step(self, batch, batch_idx):
        print("start validation")
        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        self.total_logits = torch.empty((0, self.num_labels)).to(self.device)
        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label_id = batch['label_id']
         
        # fwd
        _, logits = self.forward(input_ids, attention_mask, token_type_ids)

        self.total_logits = torch.cat(((self.total_logits.to(self.device), logits)))
        self.total_labels = torch.cat((self.total_labels.to(self.device), label_id))

        total_probs = F.softmax(self.total_logits.detach(), dim=1)

        total_maxprobs, total_preds = total_probs.max(dim = 1)
        print("max", total_maxprobs)
        print("max size", total_maxprobs.size())
        print("total preds", total_preds)
        print("total preds size", total_preds.size())
        y_pred = total_preds.cpu().numpy()
        y_true = self.total_labels.cpu().numpy()
        eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)

        self.log('val_acc', eval_score)
        print("val acc", eval_score)

        return eval_score
    
    
    def on_test_start(self):
        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        self.total_logits = torch.empty((0, self.num_labels)).to(self.device)

        return self.total_labels, self.total_logits
    
    def test_step(self, batch, batch_nb):
        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label_id = batch['label_id']
        
        _, logits = self.forward(input_ids, attention_mask, token_type_ids)
        
        self.total_logits = torch.cat(((self.total_logits.to(self.device), logits)))
        self.total_labels = torch.cat((self.total_labels.to(self.device), label_id))

        total_probs = F.softmax(self.total_logits.detach(), dim=1)
        total_maxprobs, total_preds = total_probs.max(dim = 1)

        y_pred = total_preds.cpu().numpy()
        y_true = self.total_labels.cpu().numpy()
        eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)

        
        self.log_dict({'test_acc': eval_score})
        
        return {'test_acc': eval_score}
    
    
    def predict_step(self, batch, batch_idx):
        # self.bert.eval()
        print("predict step")

        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label_ids = batch['label_id']

        # fwd
        pooled_output, _ = self.forward(input_ids, attention_mask, token_type_ids)
        # assert 1==0
        print("pooled_ouput", pooled_output)
        # print("size", pooled_output.size())  # [128, 768]

        self.total_labels = torch.cat((self.total_labels.to(self.device), label_ids))

        self.centroids = self.centroids.to(self.device)

        for i in range(len(label_ids)):
            label = label_ids[i]
            self.centroids[label] += pooled_output[i]
            # print("centroids", pooled_output[i])
        print("cebtr", self.centroids)
        
        # assert 1==0

        return {'centroids':self.centroids, 'total_labels':self.total_labels}
        # return self.centroids, self.total_labels
    
    def on_predict_end(self, ouputs):
        print("end predict centroids", self.centroids.size())

        self.total_labels = self.total_labels.cpu().numpy()
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
        param_optimizer = list(self.bert.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr = 2e-5, correct_bias=False)
        return optimizer
    
    def __build_loss(self):
        self._loss = nn.CrossEntropyLoss()


class ADBDataset(Dataset):
    def __init__(self, process_data, max_seq_len, label_list, seed):
        set_seed(seed)
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

        features = self.tokenizer.encode_plus(str(doc),
                                              add_special_tokens=True,
                                              max_length=self.max_seq_len,
                                              pad_to_max_length=True,
                                              truncation=True,
                                              return_attention_mask=True,
                                              return_tensors='pt',
                                             )        
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
    def __init__(self, data_path, dataset, batch_size, known_cls_ratio, labeled_ratio, seed, worker):
        set_seed(seed)
        self.seed = seed
        self.dataset = dataset
        self.data_path = os.path.join(data_path, self.dataset)
        self.train_data_path = f'{self.data_path}/train.tsv'
        self.val_data_path = f'{self.data_path}/dev.tsv'
        self.test_data_path = f'{self.data_path}/test.tsv'

        self.max_seq_len = max_seq_lengths[self.dataset]
        self.batch_size = batch_size
        self.known_cls_ratio = known_cls_ratio
        self.worker = worker
        self.labeled_ratio = labeled_ratio

        ####### label list
        self.all_label_list = benchmark_labels[self.dataset]
        self.n_known_cls = round(len(self.all_label_list) * self.known_cls_ratio)
        self.known_label_list = np.random.choice(np.array(self.all_label_list, dtype=str), self.n_known_cls, replace=False)
        self.known_label_list = self.known_label_list.tolist()
        print("known_label_list", self.known_label_list)

        self.num_labels = len(self.known_label_list)

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

            self.train = ADBDataset(self.train_examples, self.max_seq_len, self.label_list, self.seed) 
            self.valid = ADBDataset(self.val_examples, self.max_seq_len, self.label_list, self.seed)

        elif stage in (None, 'test'):
            # prepare
            self.test_data = pd.read_csv(self.test_data_path, delimiter="\t")

            self.test_examples = []

            for i in self.test_data.index:
                cur_label = self.test_data.loc[i]['label']
                if (cur_label in self.label_list) and (cur_label is not self.unseen_label):
                    self.test_examples.append(self.test_data.iloc[i])
                else:
                    self.test_data.loc[i]['label'] = self.unseen_label
                    self.test_examples.append(self.test_data.iloc[i])

            self.test = ADBDataset(self.test_examples, self.max_seq_len, self.label_list, self.seed)
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers= self.worker)
    
    def val_dataloader(self):
        return DataLoader(self.valid, batch_size=self.batch_size, num_workers= self.worker)
    
    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers= self.worker)
    
    def predict_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers= self.worker)
