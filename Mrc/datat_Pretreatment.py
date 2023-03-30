import json
import torch
from keras_preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader
from transformers import  BertTokenizer
from Ner.Mrc import config
import numpy as np
from  torch.utils.data import Dataset

from Ner.Mrc.config import pretrained_path


class Dateset(Dataset):
    def __init__(self,input_ids,new_start_positions,new_end_positions,type_ids,attention_masks,match_labels,types):
        self.input_ids = torch.LongTensor(input_ids)
        self.new_start_positions = torch.LongTensor(new_start_positions)
        self.new_end_positions = torch.LongTensor(new_end_positions)
        self.type_ids = torch.LongTensor(type_ids)
        self.attention_masks = torch.LongTensor(attention_masks)
        self.match_labels = torch.LongTensor(match_labels)
        self.types = types

    def __getitem__(self, index):
        return self.input_ids[index],self.new_start_positions[index],self.new_end_positions[index],self.type_ids[index],\
               self.attention_masks[index],self.match_labels[index],self.types[index]

    def __len__(self):
        return len(self.input_ids)


class pretreatment():
    def __init__(self):
        self.num = config.txt_len#取出数据集数量
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_path)

    def mrc_data(self):
        datas = []
        for i in range(1,self.num+1):
            with open(f'../../Ner_data/Ner_data{config.txt_len}/{i}.json','r',encoding='utf-8') as f:
                file = json.load(f)
            content = file['content']
            length = len(content)
            if length <= config.max_len:
                records = file['records']
                for q in config.questions:
                    start_positions = []
                    end_positions = []
                    for record in records:
                        tag=record['tag']
                        offset = record['offset']
                        span = record['span']
                        if tag == q and span not in ["0day", "0-day"]:
                            start_positions.append(offset[0])
                            end_positions.append(offset[1])
                            assert  span == content[offset[0]:offset[1]+1]
                    datas.append((content,q,config.label_des[q],start_positions,end_positions))
        print(len(datas))
        return datas

    def transformer_token(self,datas):
        ids,type_ids,start_types,end_types,new_starts,new_ends,qs = [],[],[],[],[],[],[]
        for data in datas:
            text,q,question, start_positions, end_positions = data
            qs.append(q)
            #添加question构建tokens
            t_tokens,q_tokens,tokens = [],[],[]
            for t in text:
                x = self.tokenizer.tokenize(t)
                if x == []:x = ['']
                t_tokens.append(x[0])

            for q in question:
                y = self.tokenizer.tokenize(q)
                if y == []:y = ['']
                q_tokens.append(y[0])
            tokens = ['[CLS]']+q_tokens+['[SEP]']+t_tokens+['[SEP]']
            #---------------------------------------------------------------
            #修改strat，end标签位置，构建type_ids完成句子与问题的区分
            id = self.tokenizer.convert_tokens_to_ids(tokens)
            assert len(id) <= config.pad_len
            ids.append(id)

            new_start_positions = [start_position + len(question) + 2 for start_position in start_positions]
            new_end_positions = [end_position + len(question) + 2 for end_position in end_positions]
            new_starts.append(new_start_positions)
            new_ends.append(new_end_positions)

            start_types.append([(1 if idx in new_start_positions else 0) for idx in range(len(id))])
            end_types.append([(1 if idx in new_end_positions else 0) for idx in range(len(id))])
            type_ids.append([0] * (len(question) + 2) + [1] * (len(text) + 1))

        #-------------------------------------------------------------------
        #进行批处理长度归一化
        input_ids = pad_sequences(ids, maxlen=config.pad_len, dtype='long', value=0.0, truncating="post",padding="post")
        type_ids = pad_sequences(type_ids, maxlen=config.pad_len, dtype='long', value=0, truncating="post",padding="post")
        starts = pad_sequences(start_types, maxlen=config.pad_len, dtype='long', value=0, truncating="post",padding="post")
        ends = pad_sequences(end_types, maxlen=config.pad_len, dtype='long', value=0, truncating="post",padding="post")
        attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

        match_labels = []
        for x in range(len(input_ids)):
            seq_len = len(input_ids[x])
            match_label = np.zeros([seq_len, seq_len], dtype=np.long)
            for start, end in zip(new_starts[x], new_ends[x]):
                if start >= seq_len or end >= seq_len:
                    continue
                match_label[start, end] = 1
            match_labels.append(match_label)

        assert len(input_ids) == len(type_ids) == len(attention_masks)
        for i,t,a in zip(input_ids,type_ids,attention_masks):
            assert len(i)==len(t)==len(a)==config.pad_len
        return (input_ids,starts,ends,type_ids,attention_masks,match_labels,qs)


    def loader(self,input_ids,new_start_positions,new_end_positions,type_ids,attention_masks,match_labels,types):
        dataset = Dateset(torch.LongTensor(input_ids),torch.LongTensor(new_start_positions),torch.LongTensor(new_end_positions),
                                torch.LongTensor(type_ids),torch.LongTensor(attention_masks),torch.LongTensor(match_labels),types)
        dataloader = DataLoader(dataset, batch_size=config.batch_size)
        print('dataloader load ok')
        return dataloader


if __name__ == '__main__':
    pre = pretreatment()
    datas = pre.mrc_data()
    input_ids, starts, ends, type_ids, attention_masks,match_labels,qs = pre.transformer_token(datas)
    print(pre.tokenizer.convert_ids_to_tokens(input_ids[15]),starts[15])






