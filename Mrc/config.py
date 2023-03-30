label_des = {'val':'漏洞的编号或代号','malware':'恶意软件，后门，木马的名称','attacker':'黑客，间谍，恶意攻击组织或团伙,如Apt34'}
questions = ['val','malware','attacker']
dicts = {'malware':[],'attacker':[],'val':[]}

max_len = 100
pad_len = 128
txt_len = 1757

train_size = 1000*len(questions)

num_labels=3

batch_size=4
full_fine_tuning = True
epoch_num = 20


# hyper-parameter
learning_rate = 1e-5
weight_decay = 0.01
clip_grad = 5

#loss
span_loss_candidates = "all"
loss_type = "bce"
dice_smooth = 1e-8

weight_start = 1.0
weight_end = 1.0
weight_span = 1.0  # 0.1


pretrained_path = "hfl/chinese-bert-wwm-ext"