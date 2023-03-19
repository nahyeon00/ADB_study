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


def load_model_from_experiment(args, pre):
        """Function that loads the model from an experiment folder.
        :param experiment_folder: Path to the experiment folder.
        Return:
            - Pretrained model.
        """
        # hparams_file = experiment_folder + "/hparams.yaml"
        
        # hparams_file = "/workspace/intent/newADB/lightning_logs/version_7/hparams.yaml"

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
        self.args = args

        # data
        print("argas",args.dataset)
        self.dataset = args.dataset
        self.known_cls_ratio = args.known_cls_ratio
        self.label_list = args.label_list

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
        # self.__build_loss()
        # self.criterion_boundary = BoundaryLoss(num_labels = self.num_labels, feat_dim = 768, device=self.device)
        
        # boundaryloss
        self.num_labels = self.num_labels
        self.feat_dim = 768
        self.delta = nn.Parameter(torch.randn(self.num_labels).to(self.device))
        # print("loss delta", self.delta)
        nn.init.normal_(self.delta)

        self.delta_points = []
        #self.delta = F.softplus(self.delta)
        #self.delta_points.append(self.delta)
        # print("delta points: ", self.delta_points)
        # assert 1==0

        self.unseen_label_id = self.num_labels

        self.total_labels = torch.empty(0, dtype=torch.long)
        self.total_preds = torch.empty(0,dtype=torch.long)
 
    def forward(self, input_ids, attention_mask, token_type_ids):
        pooled_output, logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return pooled_output, logits
    
    # def on_train_start(batch, batch_idx):
    #     self.delta = F.softplus(self.criterion_boundary.delta)
    #     self.delta_points.append(self.delta)
        # print("delta points: ", self.delta_points)

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
        breakpoint()
        # loss
        loss, self.delta = self.boundaryloss(pooled_output, self.centroids.to(self.device), label)
        print("ADB after loss")

        print("del", self.delta)

        # logs
        tensorboard_logs = {'train_loss': loss}
        self.delta_points.append(self.delta)
        
        self.log("train_loss", loss, on_epoch=True)

        return {'loss': loss, 'log': tensorboard_logs}
    

    def boundaryloss(self, pooled_output, centroids, labels):
        delta = F.softplus(self.delta)
        c = centroids[labels]
        d = delta[labels]
        x = pooled_output
        
        euc_dis = torch.norm(x - c,2, 1).view(-1)
        pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
        neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)
        
        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask
        
        loss = pos_loss.mean() + neg_loss.mean()

        return loss, delta 


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

        y_pred = preds.cpu().numpy()
        y_true = label_id.cpu().numpy()
        val_acc = accuracy_score(y_true, y_pred)
        eval_score = round(f1_score(y_true, y_pred, average='macro') * 100, 2)

        self.log('val_acc', val_acc)

        return val_acc

    
    def test_step(self, batch, batch_idx):
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

        return self.total_preds, self.total_labels
    
    def test_epoch_end(self, outputs):
        print("outputs", outputs[0])
        y_pred = self.total_preds.cpu().numpy()
        y_true = self.total_labels.cpu().numpy()
        print("y_pred", y_pred)
        print("y_true", y_true)
        cm = confusion_matrix(y_true, y_pred)
        print("confusion matrix!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        plot_confusion_matrix(cm, self.label_list,"confusion matrix.pdf")
        test_results = F_measure(cm)

        acc = round(accuracy_score(y_true, y_pred) * 100, 2)
        test_results['Acc'] = acc
        # self.log('test acc',acc)
        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred

        print("test results", test_results)

        tensorboard_logs = {'avg_test_acc': acc}

        self.log('test', acc)
        
        # save_results(self.args, test_results)
        
        return acc
    

    def open_classify(self, features):
        logits = euclidean_metric(features, self.centroids.to(self.device))
        probs, preds = F.softmax(logits.detach(), dim = 1).max(dim = 1)
        euc_dis = torch.norm(features - self.centroids[preds].to(self.device), 2, 1).view(-1)
        preds[euc_dis >= self.delta[preds].to(self.device)] = self.unseen_label_id

        return preds

        
    def configure_optimizers(self):
        # optimizer = AdamW(self.parameters(), lr=2e-5)
        # print("criterion", self.criterion_boundary.parameters())
        parameters = []
        for p in self.parameters():
            if p.requires_grad:
                parameters.append(p)
            else:
                print(p)

        # print("pa", self.criterion_boundary.parameters())
        # print("del", self.delta)
        # print("ppp", parameters)
        # assert 1==0
        # assert 1==0
        optimizer = torch.optim.Adam(parameters, lr=2e-5)

        return optimizer
    
    # def __build_loss(self):
    #     self.criterion_boundary = boundaryloss(num_labels = self.num_labels, feat_dim = 768, device=self.device)
    
