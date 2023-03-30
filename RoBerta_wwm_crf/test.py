import torch
from transformers import  BertTokenizer

from Ner.RoBerta_wwm_crf import config
from Ner.model.Roberta_wwm import BertLstmCrf

tokenizer = BertTokenizer.from_pretrained( "hfl/chinese-roberta-wwm-ext")
type_label = config.type_label

def format(token, label):
    dicts = {'val': [], 'malware': [], 'attacker': []}
    type = 'o'
    word = ''
    for t, l in zip(token, label):
        if l != type:
            if type != 'o': dicts[type].append(word)
            word = ''
            type = l
            word += t
        else:
            word += ' '
            word += t
    if type != 'o': dicts[type].append(word)
    return dicts

text = 'Moze恶意软件组织十分活跃'

token = []
for t in text:
    x = tokenizer.tokenize(t)
    if x == []:
        x = ['']
    token.append(x[0])

test_ids = tokenizer.convert_tokens_to_ids(token)
test_tokens_tensor = torch.tensor(test_ids)
test_tokens_tensor = test_tokens_tensor

test_masks_tensor = torch.zeros(test_tokens_tensor.shape, dtype=torch.long)
test_masks_tensor = test_masks_tensor.masked_fill(test_tokens_tensor != 0, 1)

model  = BertLstmCrf.from_pretrained(
    "bert-base-chinese",
    num_labels=config.num_labels,
    )
model.load_state_dict(torch.load('RoBerta_wwm_crf.pth'))
model.cuda()
logits = model(input_ids=test_tokens_tensor.unsqueeze(0).cuda(),attention_mask=test_masks_tensor.unsqueeze(0).cuda())[0]
preds = [type_label[i] for i in logits]

dict = format(text,preds)
print(dict)