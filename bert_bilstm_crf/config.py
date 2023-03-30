max_len = 100
txt_len = 1757
train_size = 1000

label_type = {'o':0,'val':1,'malware':2,'attacker':3,'pad':4}
type_label = {0: 'o', 1: 'val', 2: 'malware', 3: 'attacker',4: 'pad'}
num_labels=5

batch_size=8
full_fine_tuning = True
epoch_num = 20


# hyper-parameter
learning_rate = 1e-5
weight_decay = 0.01
clip_grad = 5

