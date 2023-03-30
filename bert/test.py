import torch
from transformers import  BertTokenizer
from transformers import BertForTokenClassification
from Ner.bert import config
import numpy as np
from Ner.bert.config import label_type

tokenizer = BertTokenizer.from_pretrained( "bert-base-chinese")
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

text = '针对Windows设备的CVE-2022-1111漏洞'
token = []
for t in text.lower():
    x = tokenizer.tokenize(t)
    if x == []:
        x = ['']
    token.append(x[0])

test_ids = tokenizer.convert_tokens_to_ids(token)
test_tokens_tensor = torch.tensor(test_ids)
test_tokens_tensor = test_tokens_tensor

test_masks_tensor = torch.zeros(test_tokens_tensor.shape, dtype=torch.long)
test_masks_tensor = test_masks_tensor.masked_fill(test_tokens_tensor != 0, 1)

model  = BertForTokenClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=5,
    )
model.load_state_dict(torch.load('bert.pth'))
model.cuda()
logits = model(input_ids=test_tokens_tensor.unsqueeze(0).cuda(),attention_mask=test_masks_tensor.unsqueeze(0).cuda())[0]
preds = []
for logit in logits:
    preds.extend(np.argmax(logit.detach().cpu().numpy(), axis=1))
preds = [type_label[i] for i in preds]
print(preds)
dict = format(text,preds)

print(dict)