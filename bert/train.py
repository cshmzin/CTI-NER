import torch
import os

from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import BertForTokenClassification
from Ner.bert.datat_Pretreatment import pretreatment, Dataset

from Ner.bert import config
from Ner.bert.optimizer import optimizer
import numpy as np

class Train():
    def __init__(self):
        pre = pretreatment()
        texts,labels = pre.lstm_data()
        ids = pre.transformer_token(texts)
        dataloader = Dataset()
        self.trainloader = dataloader.loader(ids[:config.train_size], labels[:config.train_size])
        self.devloader = dataloader.loader(ids[config.train_size:], labels[config.train_size:])
        #print(len(self.trainloader),len(self.devloader))
        self.model = BertForTokenClassification.from_pretrained("bert-base-chinese", num_labels=config.num_labels)
        self.model.cuda()
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.optimizer,self.scheduler = optimizer(self.model)
        self.Epochs = config.epoch_num
        self.type_label = config.type_label
        self.losses = []
        self.f1es = []

    def dev_label(self,strs,pred_tags,valid_tags):
        new_pred_tags,new_valid_tags = [],[]
        for pred_tag in pred_tags:
                if pred_tag != strs:
                    new_pred_tags.append('o')
                else:
                    new_pred_tags.append(strs)

        for valid_tag in valid_tags:
                if valid_tag != strs:
                    new_valid_tags.append('o')
                else:
                    new_valid_tags.append(strs)
        return (new_pred_tags,new_valid_tags)


    def train(self):
        self.model.train()
        losses = 0.0
        for data in self.trainloader:
            tokens_tensors, label_tensors, masks_tensors = [t for t in data]
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=tokens_tensors.cuda(), attention_mask=masks_tensors.cuda(), labels=label_tensors.cuda())
            loss = outputs[0]
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            losses += loss.item()
        avg_train_loss = losses / len(self.trainloader)
        print("Average train loss: {}".format(avg_train_loss))
        self.losses.append(avg_train_loss)

    def dev(self):
        self.model.eval()
        predictions, true_labels = [], []
        for batch in self.devloader:
            tokens_tensors, label_tensors, masks_tensors = [t for t in batch]
            with torch.no_grad():
                outputs = self.model(input_ids=tokens_tensors.cuda(), attention_mask=masks_tensors.cuda())
            for pred, label_tensor in zip(outputs[0], label_tensors):
                logit = pred.detach().cpu().numpy()  # detach的方法，将variable参数从网络中隔离开，不参与参数更新
                label_ids = label_tensor.cpu().numpy()

                predictions.extend(np.argmax(logit, axis=1))
                true_labels.append(label_ids)

        pred_tags = list(np.array(predictions).flatten())
        valid_tags = list(np.array(true_labels).flatten())
        f1_ = f1_score(pred_tags, valid_tags, average='weighted')
        print("F1-Score: {}".format(f1_))  # 传入的是具体的tag
        self.f1es.append(f1_)

    def run(self):
        if os.path.exists('bert.pth'):self.model.load_state_dict(torch.load('bert.pth'))
        for epoch in range(self.Epochs):
            self.train()
            self.dev()
        torch.save(self.model.state_dict(), 'bert.pth')
        print(self.losses)
        print(self.f1es)

    def dev_result(self):
        if os.path.exists('bert.pth'): self.model.load_state_dict(torch.load('bert.pth'))
        self.model.eval()
        predictions, true_labels = [], []
        for batch in self.devloader:
            tokens_tensors, label_tensors, masks_tensors = [t for t in batch]
            with torch.no_grad():
                outputs = self.model(input_ids=tokens_tensors.cuda(), attention_mask=masks_tensors.cuda())
            for pred, label_tensor in zip(outputs[0], label_tensors):
                logit = pred.detach().cpu().numpy()  # detach的方法，将variable参数从网络中隔离开，不参与参数更新
                label_ids = label_tensor.cpu().numpy()

                predictions.extend(np.argmax(logit, axis=1))
                true_labels.append(label_ids)

        pred_tags = list(np.array(predictions).flatten())
        valid_tags = list(np.array(true_labels).flatten())

        pred_tags = [p for p, l in zip(pred_tags, valid_tags) if self.type_label[l] != "pad"]
        valid_tags = [l for l in valid_tags if self.type_label[l] != "pad"]

        f1_ = f1_score(pred_tags, valid_tags, average='weighted')
        precision = precision_score(pred_tags, valid_tags, average='weighted')
        recall = recall_score(pred_tags, valid_tags, average='weighted')
        print("Validation F1-Score: {}".format(f1_))
        print("Validation precision_score: {}".format(precision))
        print("Validation recall_score: {}".format(recall))
        #---------------------------------------------------------------------------------------------------
        val_pred,val_valid = self.dev_label(1,pred_tags,valid_tags)
        malware_pred, malware_valid = self.dev_label(2, pred_tags, valid_tags)
        attacker_pred, attacker_valid = self.dev_label(3, pred_tags, valid_tags)
        #---------------------------------------------------------------------------------------------------

        val_f1_ = f1_score(val_pred,val_valid, average='weighted')
        val_precision = precision_score(val_pred,val_valid, average='weighted')
        val_recall = recall_score(val_pred,val_valid, average='weighted')
        print('val:')
        print("Validation F1-Score: {}".format(val_f1_))
        print("Validation precision_score: {}".format(val_precision))
        print("Validation recall_score: {}".format(val_recall))

        #----------------------------------------------------------------------------------------------------
        malware_f1_ = f1_score(malware_pred,malware_valid,average='weighted')
        malware_precision = precision_score(malware_pred,malware_valid, average='weighted')
        malware_recall = recall_score(malware_pred,malware_valid, average='weighted')
        print('malware:')
        print("Validation F1-Score: {}".format(malware_f1_))
        print("Validation precision_score: {}".format(malware_precision))
        print("Validation recall_score: {}".format(malware_recall))

        #-----------------------------------------------------------------------------------------------------
        attacker_f1_ = f1_score(attacker_pred,attacker_valid, average='weighted')
        attacker_precision = precision_score(attacker_pred,attacker_valid, average='weighted')
        attacker_recall = recall_score(attacker_pred,attacker_valid, average='weighted')
        print('attacker:')
        print("Validation F1-Score: {}".format(attacker_f1_))
        print("Validation precision_score: {}".format(attacker_precision))
        print("Validation recall_score: {}".format(attacker_recall))


if __name__ == '__main__':
    train = Train()
    #train.run()
    train.dev_result()
