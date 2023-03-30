import re

import torch
from transformers import  BertTokenizer
from torch.utils.data import DataLoader
from Ner.CTIMrc import config
from Ner.CTIMrc.config import  pretrained_path
from Ner.model.CTImrc import Bert_Mrc
from  torch.utils.data import Dataset
from keras_preprocessing.sequence import pad_sequences

with open('../dicts/malwares.txt', 'r', encoding='utf-8') as f:
    mals = f.readlines()

with open('../dicts/malware_reverse.txt', 'r', encoding='utf-8') as f2:
    mals_re = f2.readlines()

with open('../dicts/attackers.txt', 'r', encoding='utf-8') as f:
    atts = f.readlines()

with open('../dicts/attacker_reverse.txt', 'r', encoding='utf-8') as f:
    atts_re = f.readlines()


class Dateset(Dataset):
    def __init__(self,input_ids,type_ids,attention_masks,types,kg_matrixs):
        self.input_ids = torch.LongTensor(input_ids)
        self.type_ids = torch.LongTensor(type_ids)
        self.attention_masks = torch.LongTensor(attention_masks)
        self.types = types
        self.kg_matrixs = kg_matrixs

    def __getitem__(self, index):
        return self.input_ids[index],self.type_ids[index],self.attention_masks[index],self.types[index],self.kg_matrixs[index]

    def __len__(self):
        return len(self.input_ids)


def Knowledge_matrix(content, q):
    content = content.lower()
    matrix = [1.0 for c in content]
    if q == 'malware':
        for m in mals:
            m = re.sub(r'\n', '', m)
            mxs = re.finditer(fr'{m.lower()}', content)
            for x in mxs:
                for i in range(x.span()[0], x.span()[1]):
                    matrix[i] = 2.0

        for m in mals_re:
            m = re.sub(r'\n', '', m)
            mxs = re.finditer(fr'{m.lower()}', content)
            for x in mxs:
                for i in range(x.span()[0], x.span()[1]):
                    matrix[i] = 0.1

    elif q == 'attacker':
        for a in atts:
            a = re.sub(r'\n', '', a)
            axs = re.finditer(fr'{a.lower()}', content)
            for x in axs:
                for i in range(x.span()[0], x.span()[1]):
                    matrix[i] = 2.0

        for a in atts_re:
            a = re.sub(r'\n', '', a)
            axs = re.finditer(fr'{a.lower()}', content)
            for x in axs:
                for i in range(x.span()[0], x.span()[1]):
                    matrix[i] = 0.1

    return matrix

def extract_flat_spans(start_pred, end_pred, match_pred, label_mask, pseudo_tag):
    bmes_labels = ["O"] * len(start_pred)
    start_positions = [idx for idx, tmp in enumerate(start_pred) if tmp and label_mask[idx]]
    end_positions = [idx for idx, tmp in enumerate(end_pred) if tmp and label_mask[idx]]

    for start_item in start_positions:
        bmes_labels[start_item] = f"B"
    for end_item in end_positions:
        bmes_labels[end_item] = f"I"

    for tmp_start in start_positions:
        tmp_end = [tmp for tmp in end_positions if tmp >= tmp_start]
        if len(tmp_end) == 0:
            continue
        else:
            tmp_end = min(tmp_end)
        if match_pred[tmp_start][tmp_end]:
            if tmp_start != tmp_end:
                for i in range(tmp_start + 1, tmp_end):
                    bmes_labels[i] = f"I"
            else:
                bmes_labels[tmp_end] = f"B"
    return bmes_labels


def extract_flat_spans_batch(start_pred, end_pred, match_pred, label_mask, pseudo_tag):
    batch_label = []
    B, length = start_pred.size()
    for i in range(B):
        temp_start_pred, temp_end_pred, temp_match_pred, temp_label_mask, temp_pseudo_tag = \
            start_pred[i, :], end_pred[i, :], match_pred[i, :, :], label_mask[i, :], pseudo_tag[i]
        temp_bio_label = extract_flat_spans(
            temp_start_pred,
            temp_end_pred,
            temp_match_pred,
            temp_label_mask,
            temp_pseudo_tag
        )
        batch_label.append(temp_bio_label)
    return batch_label

def make_dict(predic,test,ques):
    mask_predic = predic[len(ques)+2:]
    word = ''
    words = []
    for t,p in zip(test,mask_predic):
        if p == 'B':
            word = t
        elif p == 'I' and word != '':
            word += t
        else:
            if word != '':
                words.append(word)
            word = ''
    return words


def ner_test(texts):
    tokenizer = BertTokenizer.from_pretrained(pretrained_path)
    label_des = config.label_des
    questions = config.questions

    model = Bert_Mrc.from_pretrained(pretrained_path)
    model.load_state_dict(torch.load('Mrc.pth'))

    dicts = {'val':[],'malware':[],'attacker':[]}

    input_id = []
    token_type_ids =[]
    types = []
    kg_matrixs = []

    for text in texts:
        for q in questions:
            t_tokens, q_tokens = [], []
            ques = label_des[q]
            for t in text:
                x = tokenizer.tokenize(t)
                if x == []: x = ['']
                t_tokens.append(x[0])
            for que in ques:
                y = tokenizer.tokenize(que)
                if y == []: y = ['']
                q_tokens.append(y[0])

            kg_matrix = Knowledge_matrix(text, q)
            kg_matrix = [1.0] + len(q_tokens)*[1.0] + [1.0] + kg_matrix + [1.0]
            kg_matrixs.append(kg_matrix)

            tokens = ['[CLS]'] + q_tokens + ['[SEP]'] + t_tokens + ['[SEP]']
            id = tokenizer.convert_tokens_to_ids(tokens)
            assert len(id) <= config.pad_len
            type_id = [0] * (len(ques) + 2) + [1] * (len(text) + 1)
            input_id.append(id)
            token_type_ids.append(type_id)
            types.append(q)

    kg_matrixs = pad_sequences(kg_matrixs,maxlen=config.pad_len,dtype='float32',value=1.0,truncating="post",padding="post")

    input_ids = pad_sequences(input_id, maxlen=config.pad_len, dtype='long', value=0.0, truncating="post",
                              padding="post")
    type_ids = pad_sequences(token_type_ids, maxlen=config.pad_len, dtype='long', value=0, truncating="post",
                             padding="post")
    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

    dataset = Dateset(input_ids, type_ids, attention_masks,types,kg_matrixs)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)

    for batch in dataloader:
        tokens_tensors, type_ids_tensors, masks_tensors, type_ ,kg_matrixs= [t for t in batch]
        with torch.no_grad():
            start_logits, end_logits, span_logits = model(input_ids=tokens_tensors,token_type_ids=type_ids_tensors,attention_mask=masks_tensors,kg_matrixs = kg_matrixs)
        start_preds, end_preds, span_pred = start_logits > 0, end_logits > 0, span_logits > 0
        predic = extract_flat_spans_batch(start_preds,end_preds,span_pred,type_ids_tensors,type_)
        print(len(predic))

    len_texts = len(texts)
    len_question = len(config.questions)
    pres = []
    for i in range(len_texts):
        pres.append(predic[i*len_question:(i+1)*len_question])

    for i in range(len_texts):
        text = texts[i]
        for l in range(len_question):
            q = questions[l]
            pre = pres[i][l]
            dicts[q] += make_dict(pre, text, label_des[q])
    return dicts

if __name__ == '__main__':
    # texts = ["一种名为mirai的DDos僵尸网络"]
    texts = ["Moze恶意软件组织十分活跃"]
    print(ner_test(texts))
