import torch
from seqeval.metrics import f1_score, precision_score, recall_score,accuracy_score
import os
from torch.nn.modules import BCEWithLogitsLoss

from Ner.Mrc.config import pretrained_path
from Ner.Mrc.datat_Pretreatment import pretreatment
from Ner.Mrc.loss import DiceLoss
from Ner.model.Bert_mrc import Bert_Mrc
from Ner.Mrc import config
from Ner.Mrc.optimizer import optimizer
import numpy as np

class Train():
    def __init__(self):
        pre = pretreatment()
        datas = pre.mrc_data()
        input_ids, starts, ends, type_ids, attention_masks,match_labels,qs = pre.transformer_token(datas)
        print(len(input_ids))
        self.trainloader = pre.loader(input_ids[:config.train_size], starts[:config.train_size], ends[:config.train_size],
                                      type_ids[:config.train_size], attention_masks[:config.train_size],match_labels[:config.train_size],qs[:config.train_size])
        self.devloader = pre.loader(input_ids[config.train_size:], starts[config.train_size:], ends[config.train_size:],
                                    type_ids[config.train_size:], attention_masks[config.train_size:],match_labels[config.train_size:],qs[config.train_size:])

        #print(len(self.trainloader),len(self.devloader))
        self.model = Bert_Mrc.from_pretrained(pretrained_path)
        self.model.cuda()
        #self.model.cuda()
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        self.optimizer,self.scheduler = optimizer(self.model)
        self.Epochs = config.epoch_num
        self.losses = []
        self.f1es = []

    def dev_label(self,strs,pred_tags,valid_tags):
        new_pred_tags,new_valid_tags = [],[]
        for pred_tag in pred_tags:
            new_pred_tag = []
            for p in pred_tag:
                if p != f"B-{strs}" and p != f"I-{strs}":
                    new_pred_tag.append('o')
                else:
                    new_pred_tag.append(p)
            new_pred_tags.append(new_pred_tag)

        for valid_tag in valid_tags:
            new_valid_tag = []
            for p in valid_tag:
                if p != f"B-{strs}" and\
                        p != f"I-{strs}":
                    new_valid_tag.append('o')
                else:
                    new_valid_tag.append(p)
            new_valid_tags.append(new_valid_tag)

        return (new_pred_tags,new_valid_tags)

    def extract_flat_spans(self,start_pred, end_pred, match_pred, label_mask, pseudo_tag):
        bmes_labels = ["O"] * len(start_pred)
        start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and label_mask[idx]]
        end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and label_mask[idx]]

        for start_item in start_positions:
            bmes_labels[start_item] = f"B-{pseudo_tag}"
        for end_item in end_positions:
            bmes_labels[end_item] = f"I-{pseudo_tag}"

        for tmp_start in start_positions:
            tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
            if len(tmp_end) == 0:
                continue
            else:
                tmp_end = min(tmp_end)
            if match_pred[tmp_start][tmp_end]:
                if tmp_start != tmp_end:
                    for i in range(tmp_start + 1, tmp_end):
                        bmes_labels[i] = f"I-{pseudo_tag}"
                else:
                    bmes_labels[tmp_end] = f"B-{pseudo_tag}"
        return bmes_labels

    def extract_flat_spans_batch(self,start_pred, end_pred, match_pred, label_mask, pseudo_tag):
        batch_label = []
        B, length = start_pred.size()
        for i in range(B):
            temp_start_pred, temp_end_pred, temp_match_pred, temp_label_mask, temp_pseudo_tag = \
                start_pred[i, :], end_pred[i, :], match_pred[i, :, :], label_mask[i, :], pseudo_tag[i]
            temp_bio_label = self.extract_flat_spans(
                temp_start_pred,
                temp_end_pred,
                temp_match_pred,
                temp_label_mask,
                temp_pseudo_tag
            )
            batch_label.append(temp_bio_label)
        return batch_label

    def compute_loss(self,start_logits, end_logits, span_logits,
                     start_labels, end_labels, match_labels, start_label_mask, end_label_mask):
        batch_size, seq_len = start_logits.size()

        start_float_label_mask = start_label_mask.view(-1).float()
        end_float_label_mask = end_label_mask.view(-1).float()
        match_label_row_mask = start_label_mask.bool().unsqueeze(-1).expand(-1, -1, seq_len)
        match_label_col_mask = end_label_mask.bool().unsqueeze(-2).expand(-1, seq_len, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = torch.triu(match_label_mask, 0)  # start should be less equal to end

        if config.span_loss_candidates == "all":
            # naive mask
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        else:
            # use only pred or golden start/end to compute match loss
            start_preds = start_logits > 0
            end_preds = end_logits > 0
            if config.span_loss_candidates == "gold":
                match_candidates = ((start_labels.unsqueeze(-1).expand(-1, -1, seq_len) > 0)
                                    & (end_labels.unsqueeze(-2).expand(-1, seq_len, -1) > 0))
            else:
                match_candidates = torch.logical_or(
                    (start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)),
                    (start_labels.unsqueeze(-1).expand(-1, -1, seq_len)
                     & end_labels.unsqueeze(-2).expand(-1, seq_len, -1))
                )
            match_label_mask = match_label_mask & match_candidates
            float_match_label_mask = match_label_mask.view(batch_size, -1).float()
        if config.loss_type == "bce":
            bce_loss = BCEWithLogitsLoss(reduction="none")
            start_loss = bce_loss(start_logits.view(-1), start_labels.view(-1).float())
            start_loss = (start_loss * start_float_label_mask).sum() / start_float_label_mask.sum()
            end_loss = bce_loss(end_logits.view(-1), end_labels.view(-1).float())
            end_loss = (end_loss * end_float_label_mask).sum() / end_float_label_mask.sum()
            match_loss = bce_loss(span_logits.view(batch_size, -1), match_labels.view(batch_size, -1).float())
            match_loss = match_loss * float_match_label_mask
            match_loss = match_loss.sum() / (float_match_label_mask.sum() + 1e-10)
        else:
            dice_loss = DiceLoss(with_logits=True, smooth=config.dice_smooth)
            start_loss = dice_loss(start_logits, start_labels.float(), start_float_label_mask)
            end_loss = dice_loss(end_logits, end_labels.float(), end_float_label_mask)
            match_loss = dice_loss(span_logits, match_labels.float(), float_match_label_mask)

        return start_loss, end_loss, match_loss

    def train(self):
        self.model.train()
        losses = 0.0
        for data in self.trainloader:
            tokens_tensors, starts_tensors,ends_tensors,type_ids_tensors, masks_tensors,match_labels,type_ = [t for t in data]
            self.optimizer.zero_grad()
            start_logits, end_logits, span_logits = self.model(input_ids=tokens_tensors.cuda(),token_type_ids=type_ids_tensors.cuda(),attention_mask=masks_tensors.cuda())
            start_loss, end_loss, match_loss = self.compute_loss(
                start_logits=start_logits.cpu(),
                end_logits=end_logits.cpu(),
                span_logits=span_logits.cpu(),
                start_labels=starts_tensors,
                end_labels=ends_tensors,
                match_labels=match_labels,
                start_label_mask=type_ids_tensors,
                end_label_mask=type_ids_tensors
                )
            loss = config.weight_start * start_loss + config.weight_end * end_loss + config.weight_span * match_loss
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            losses += loss.item()
        avg_train_loss = losses / len(self.trainloader)
        print("Average train loss: {}".format(avg_train_loss))
        self.losses.append(avg_train_loss)

    def dev(self):
        self.model.eval()
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        for batch in self.devloader:
            tokens_tensors, starts_tensors,ends_tensors,type_ids_tensors, masks_tensors,match_labels,type_ = [t for t in batch]
            with torch.no_grad():
                start_logits, end_logits, span_logits = self.model(input_ids=tokens_tensors.cuda(),token_type_ids=type_ids_tensors.cuda(),attention_mask=masks_tensors.cuda())
            start_preds, end_preds, span_pred = start_logits.cpu() > 0, end_logits.cpu() > 0, span_logits.cpu()>0
            active_labels = self.extract_flat_spans_batch(start_pred=starts_tensors,
                                                     end_pred=ends_tensors,
                                                     match_pred=match_labels,
                                                     label_mask=type_ids_tensors,
                                                     pseudo_tag=type_
                                                     )
            predic = self.extract_flat_spans_batch(start_pred=start_preds,
                                              end_pred=end_preds,
                                              match_pred=span_pred,
                                              label_mask=type_ids_tensors,
                                              pseudo_tag=type_
                                            )
            labels_all = np.append(labels_all, active_labels)
            predict_all = np.append(predict_all, predic)

        true_label = labels_all
        predict_label = predict_all

        f1 = f1_score([list(true_label)], [list(predict_label)],average='macro')
        acc = accuracy_score([list(true_label)], [list(predict_label)])
        print("Validation F1-Score: {}".format(f1))
        print("Validation Acc: {}".format(acc))
        self.f1es.append(f1)


    def Print_result(self,label,pred,valid):
        f1_ = f1_score(pred,valid)
        precision = precision_score(pred,valid)
        recall = recall_score(pred,valid)
        acc = accuracy_score(pred,valid)

        print(f'{label}:')
        print("Validation F1-Score: {}".format(f1_))
        print("Validation precision_score: {}".format(precision))
        print("Validation recall_score: {}".format(recall))
        print("Validation acc: {}".format(acc))

    def dev_result(self):
        if os.path.exists('Mrc.pth'):
            self.model.load_state_dict(torch.load('Mrc.pth'))
        self.model.eval()
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        for batch in self.devloader:
            tokens_tensors, starts_tensors,ends_tensors,type_ids_tensors, masks_tensors,match_labels,type_ = [t for t in batch]
            with torch.no_grad():
                start_logits, end_logits, span_logits = self.model(input_ids=tokens_tensors.cuda(),token_type_ids=type_ids_tensors.cuda(),attention_mask=masks_tensors.cuda())
            start_preds, end_preds, span_pred = start_logits.cpu() > 0, end_logits.cpu() > 0, span_logits.cpu()>0
            active_labels = self.extract_flat_spans_batch(start_pred=starts_tensors,
                                                     end_pred=ends_tensors,
                                                     match_pred=match_labels,
                                                     label_mask=type_ids_tensors,
                                                     pseudo_tag=type_
                                                     )
            predic = self.extract_flat_spans_batch(start_pred=start_preds,
                                              end_pred=end_preds,
                                              match_pred=span_pred,
                                              label_mask=type_ids_tensors,
                                              pseudo_tag=type_
                                            )
            labels_all = np.append(labels_all, active_labels)
            predict_all = np.append(predict_all, predic)

        true_label = labels_all
        predict_label = predict_all

        pred_tags, valid_tags = [list(true_label)],[list(predict_label)]

        f1_ = f1_score(pred_tags, valid_tags,average='macro')
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

    def run(self):
        if os.path.exists('Mrc.pth'):
            self.model.load_state_dict(torch.load('Mrc.pth'))
            print('use load_state')
        for epoch in range(self.Epochs):
            self.train()
            self.dev()
            self.dev_result()
        torch.save(self.model.state_dict(), 'Mrc.pth')
        print(self.losses)
        print(self.f1es)


if __name__ == '__main__':
    train = Train()
    # train.run()
    train.dev_result()