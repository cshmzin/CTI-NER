import json
import torch
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader
from transformers import  BertTokenizer

from Ner.RoBerta_wwm_crf import config


class pretreatment():
    def __init__(self):
        self.num = config.txt_len#取出数据集数量
        self.dict = config.label_type
        self.tokenizer = BertTokenizer.from_pretrained( "hfl/chinese-roberta-wwm-ext")

    def lstm_data(self):
        labels = []
        texts = []
        for i in range(1,self.num+1):
            with open(f'../../Ner_data/Ner_data{config.txt_len}/{i}.json','r',encoding='utf-8') as f:
                file = json.load(f)
            content = file['content']
            length = len(content)
            if length <= config.max_len:
                label = [0]*length
                records = file['records']

                for record in records:
                    tag=record['tag']
                    offset = record['offset']
                    span = record['span']
                    l = self.dict[tag]
                    for d in range(offset[0],offset[1]+1):
                        label[d] = l
                    assert  span == content[offset[0]:offset[1]+1]

                labels.append(label)
                texts.append(content)
        print(len(texts))
        return (texts,labels)

    def transformer_token(self,texts):
        tokens = []
        for text in texts:
            token = []
            for t in text:
                x = self.tokenizer.tokenize(t)
                if x == []:
                    x = ['']
                token.append(x[0])
            tokens.append(token)
        ids = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]
        return ids

class Dataset():
    def __init__(self):
        self.label_type = config.label_type

    def pad(self,ids,labels):

        input_ids = pad_sequences(ids, maxlen=config.max_len, dtype='long', value=0.0, truncating="post", padding="post")
        tags = pad_sequences(labels, maxlen=config.max_len, value=self.label_type["pad"], padding="post", dtype="long", truncating="post")
        attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]
        return (input_ids,tags,attention_masks)

    def loader(self,ids,labels):
        input_ids,tags,attention_masks = self.pad(ids,labels)
        dataset = TensorDataset(torch.LongTensor(input_ids),torch.LongTensor(tags),torch.LongTensor(attention_masks))
        dataloader = DataLoader(dataset, batch_size=config.batch_size)
        print('dataloader load ok')
        return dataloader


if __name__ == '__main__':
    pre = pretreatment()
    texts,labels = pre.lstm_data()
    ids = pre.transformer_token(texts)
    print(texts[0],ids[0])





