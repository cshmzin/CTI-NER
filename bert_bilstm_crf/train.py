import torch
from seqeval.metrics import f1_score as f1
from seqeval.metrics import precision_score,recall_score,accuracy_score
import os

from Ner.bert_bilstm_crf.datat_Pretreatment import pretreatment, Dataset
from Ner.model.Bert_lstm_crf import BertLstmCrf

from Ner.bert_bilstm_crf import config
from Ner.bert_bilstm_crf.optimizer import optimizer


class Train():
    def __init__(self):
        pre = pretreatment()
        texts,labels = pre.lstm_data()
        ids = pre.transformer_token(texts)
        dataloader = Dataset()
        self.trainloader = dataloader.loader(ids[:config.train_size], labels[:config.train_size])
        self.devloader = dataloader.loader(ids[config.train_size:], labels[config.train_size:])
        #print(len(self.trainloader),len(self.devloader))
        self.model = BertLstmCrf.from_pretrained("bert-base-chinese", num_labels=config.num_labels)
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
            new_pred_tag = []
            for p in pred_tag:
                if p != strs:
                    new_pred_tag.append('o')
                else:
                    new_pred_tag.append(strs)
            new_pred_tags.append(new_pred_tag)

        for valid_tag in valid_tags:
            new_valid_tag = []
            for p in valid_tag:
                if p != strs:
                    new_valid_tag.append('o')
                else:
                    new_valid_tag.append(strs)
            new_valid_tags.append(new_valid_tag)

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
                outputs = self.model(input_ids=tokens_tensors.cuda(), attention_mask=masks_tensors.cuda(), labels=label_tensors.cuda())
            logits = outputs[1]
            label_ids = label_tensors.cpu().numpy()

            predictions.extend(logits)
            true_labels.extend(list(label_ids))
        pred_tags = [[self.type_label[p_i] for p, l in zip(predictions, true_labels)
                      for p_i, l_i in zip(p, l) if self.type_label[l_i] != "pad"]]
        valid_tags = [[self.type_label[l_i] for l in true_labels
                       for l_i in l if self.type_label[l_i] != "pad"]]
        f1_ = f1(pred_tags, valid_tags,average='macro')
        acc = accuracy_score(pred_tags, valid_tags)
        print("Validation F1-Score: {}".format(f1_))
        print("Validation Acc: {}".format(acc))
        self.f1es.append(f1_)

    def run(self):
        if os.path.exists('bert_bilstm_crf.pth'):self.model.load_state_dict(torch.load('bert_bilstm_crf.pth'))
        for epoch in range(self.Epochs):
            self.train()
            self.dev()
        torch.save(self.model.state_dict(), 'bert_bilstm_crf.pth')
        print(self.losses)
        print(self.f1es)

    def Print_result(self,label,pred,valid):
        f1_ = f1(pred,valid,average='macro')
        precision = precision_score(pred,valid)
        recall = recall_score(pred,valid)
        acc = accuracy_score(pred,valid)

        print(f'{label}:')
        print("Validation F1-Score: {}".format(f1_))
        print("Validation precision_score: {}".format(precision))
        print("Validation recall_score: {}".format(recall))
        print("Validation acc: {}".format(acc))



    def dev_result(self):
        if os.path.exists('bert_bilstm_crf.pth'): self.model.load_state_dict(torch.load('bert_bilstm_crf.pth'))
        self.model.eval()
        predictions, true_labels = [], []

        for batch in self.devloader:
            tokens_tensors, label_tensors, masks_tensors = [t for t in batch]
            with torch.no_grad():
                outputs = self.model(input_ids=tokens_tensors.cuda(), attention_mask=masks_tensors.cuda(), labels=label_tensors.cuda())
            logits = outputs[1]
            label_ids = label_tensors.cpu().numpy()

            predictions.extend(logits)
            true_labels.extend(list(label_ids))
        pred_tags = [[self.type_label[p_i] for p, l in zip(predictions, true_labels)
                      for p_i, l_i in zip(p, l) if self.type_label[l_i] != "pad"]]
        valid_tags = [[self.type_label[l_i] for l in true_labels
                       for l_i in l if self.type_label[l_i] != "pad"]]


        f1_ = f1(pred_tags, valid_tags,average='macro')
        precision = precision_score(pred_tags, valid_tags)
        recall = recall_score(pred_tags, valid_tags)
        print("Validation F1-Score: {}".format(f1_))
        print("Validation precision_score: {}".format(precision))
        print("Validation recall_score: {}".format(recall))
        #---------------------------------------------------------------------------------------------------
        vul_pred,vul_valid = self.dev_label('val',pred_tags,valid_tags)
        malware_pred, malware_valid = self.dev_label('malware', pred_tags, valid_tags)
        attacker_pred, attacker_valid = self.dev_label('attacker', pred_tags, valid_tags)
        #location_pred, location_valid = self.dev_label('location', pred_tags, valid_tags)
        #---------------------------------------------------------------------------------------------------
        self.Print_result('vul',vul_pred,vul_valid)
        self.Print_result('malware', malware_pred, malware_valid)
        self.Print_result('attacker', attacker_pred, attacker_valid)
        #self.Print_result('location', location_pred, location_valid)



if __name__ == '__main__':
    train = Train()
    # train.run()
    train.dev_result()
