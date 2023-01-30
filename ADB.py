from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, AdamW
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
import yaml
from argparse import ArgumentParser, Namespace
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

from boundaryloss import *
from model import *
from data_info import * 
from utils import *


def load_model_from_experiment(args):
        """Function that loads the model from an experiment folder.
        :param experiment_folder: Path to the experiment folder.
        Return:
            - Pretrained model.
        """
        # hparams_file = experiment_folder + "/hparams.yaml"
        
        # hparams_file = "/workspace/intent/newADB/lightning_logs/version_0/hparams.yaml"

        # hparams = yaml.load(open(hparams_file).read(), Loader=yaml.FullLoader)

        checkpoints = [
            file
            for file in os.listdir("/workspace/intent/newADB/checkpoints/")
            if file.endswith(".ckpt")
        ]
        # checkpoint_path = ckpt_path + checkpoints[-1]
        # model = BERTfeature.load_from_checkpoint(
        #     checkpoint_path, hparams=Namespace(**hparams)
        # )
        print("name", checkpoints[-1])
        checkpoint_path = args.ckpt_path + "/checkpoints/" + checkpoints[-1]
        print("checkpoint", checkpoint_path)
        model = BERTfeature.load_from_checkpoint(checkpoint_path, args =args)
        return model

class BERTADB(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.automatic_optimization = False
        self.args = args

        # data
        print("argas",args.dataset)
        self.dataset = args.dataset
        self.known_cls_ratio = args.known_cls_ratio

        self.num_labels = args.num_labels

        # 모델, 중심 불러오기
        # self.model = load_model_from_experiment(args)
        checkpoints = [
            file
            for file in os.listdir("/workspace/intent/newADB/checkpoints/")
            if file.endswith(".ckpt")
        ]
        print("chekc", checkpoints)
        checkpoint_path = args.ckpt_path + "/checkpoints/" + checkpoints[0]
        print("checkpoint", checkpoint_path)
        self.model = BERTfeature.load_from_checkpoint(checkpoint_path, args=args)
        
        file_name = f'centroids_{self.dataset}_{self.known_cls_ratio}.npy'
        self.centroids = np.load(os.path.join(args.centroids, file_name))
        self.centroids = torch.from_numpy(self.centroids).to(self.device)
        # self.centroids = np.load('/workspace/TEXTOIR/open_intent_detection/centroids.npy')
        # self.centroids = torch.from_numpy(self.centroids).to(self.device)
        
        
        print("centroids", self.centroids)
        
        
        
        # delta 초기화
        # delta 값
        
        self.criterion_boundary = BoundaryLoss(num_labels = self.num_labels, feat_dim = 768, device=self.device)
        
        self.delta_points = []
        self.delta = F.softplus(self.criterion_boundary.delta)
        self.delta_points.append(self.delta)
        print("delta points: ", self.delta_points)

        self.total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        self.total_preds = torch.empty(0,dtype=torch.long).to(self.device)

        self.unseen_label_id = self.num_labels


    def forward(self, input_ids, attention_mask, token_type_ids):
        pooled_output, logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        print("ADB forward pooledoutput: ", pooled_output.shape)
        print("ADB forward logit: ", logits.shape)

        return pooled_output, logits


    def training_step(self, batch, batch_idx):
        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label = batch['label_id']
        
        # fwd
        print("ADB forward")
        ###################
        pooled_output, _ = self.forward(input_ids, attention_mask, token_type_ids)  # feature
        print("before boundary loss")
        # loss
        loss, self.delta = self.criterion_boundary(pooled_output, self.centroids.to(self.device), label)
        print("ADB after loss")
        
        # logs
        tensorboard_logs = {'train_loss': loss}
        self.delta_points.append(self.delta)
        
        self.log("train_loss", loss, on_epoch=True)

        return {'loss': loss, 'log': tensorboard_logs}

    
    def on_validation_start(self):
        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        self.total_preds = torch.empty(0,dtype=torch.long).to(self.device)

        return self.total_labels, self.total_preds
    

    def validation_step(self, batch, batch_idx):
        print("validation step")
        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label_id = batch['label_id']
         
        # fwd
        pooled_output, _ = self.forward(input_ids, attention_mask, token_type_ids)
        
        # loss
        preds = self.open_classify(pooled_output)
        
        self.total_preds = torch.cat((self.total_preds.to(self.device), preds))
        self.total_labels = torch.cat((self.total_labels.to(self.device), label_id))


        y_pred = self.total_preds.cpu().numpy()
        y_true = self.total_labels.cpu().numpy()
        eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)
        # loss
        # loss_val = self._loss(logits, label.squeeze(-1))

        self.log('val_acc', eval_score)
        # print("val acc", val_acc)

        # return {'val_loss': loss_val, 'val_acc': val_acc}
        return eval_score
        
        # eval_score = round(f1_score(y_true, y_pred, average='macro') * 100, 2)
        
        # self.log('eval_score', self.total_preds)

        # return {'total_preds':self.total_preds, 'total_labels':self.total_labels}
        # return eval_score

    
    # def validation_step_end(self):
    #     print("validation end")
    #     y_pred = self.total_preds.cpu().numpy()
    #     y_true = self.total_labels.cpu().numpy()
        
    #     eval_score = round(f1_score(y_true, y_pred, average='macro') * 100, 2)
    #     # self.log('eval_score', eval_score)
    #     # tensorboard_logs = {'val_loss': avg_loss,'avg_val_acc':avg_val_acc}
    #     return eval_score
    
    def test_step(self, batch, batch_nb):
        print("test step")
        # # batch
        # input_ids, attention_mask, token_type_ids, label = batch
        # batch
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        label_id = batch['label_id']
        
        pooled_output, _ = self.forward(input_ids, attention_mask, token_type_ids)
        
        preds = self.open_classify(pooled_output)
        
        self.total_preds = torch.cat((self.total_preds.to(self.device), preds))
        self.total_labels = torch.cat((self.total_labels.to(self.device), label_id))
        self.log('preds', self.total_preds)
        
    #     return {'total_preds':self.total_preds, 'total_labels':self.total_labels}
    
    # def on_test_end(self):
        print("on test end")
        y_pred = self.total_preds.cpu().numpy()
        y_true = self.total_labels.cpu().numpy()
        print("y_pred", y_pred)
        print("y_true", y_true)

        cm = confusion_matrix(y_true, y_pred)
        print("confusion matrix!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        test_results = F_measure(cm)

        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        test_results['Acc'] = acc
        # self.log('test acc',acc)
        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred
        print("test results", test_results)

        tensorboard_logs = {'avg_test_acc': acc}
        
        save_results(self.args, test_results)
        
        return acc
    

    def open_classify(self, features):
        # print("feature device", features.device)  # gpu
        # print("centroid device", self.centroids.device)  # cpu
        logits = euclidean_metric(features, self.centroids.to(self.device))
        probs, preds = F.softmax(logits.detach(), dim = 1).max(dim = 1)
        # print("feature", features.device)
        # print("centroid device", self.centroids.device)  # cpu
        euc_dis = torch.norm(features - self.centroids[preds].to(self.device), 2, 1).view(-1)
        # print('eud', euc_dis.device)
        # print("del", self.delta[preds].device)
        preds[euc_dis >= self.delta[preds].to(self.device)] = self.unseen_label_id

        return preds

        
    def class_count(labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num

    def configure_optimizers(self):
        # optimizer = AdamW(self.parameters(), lr=2e-5)
        optimizer = torch.optim.Adam(self.criterion_boundary.parameters(), lr=2e-5)
        return optimizer
    
