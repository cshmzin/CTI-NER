from transformers import BertPreTrainedModel,BertModel
from torchcrf import CRF
import torch.nn as nn
class BertLstmCrf(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config,need_bilstm = True,rnn_dim = 64):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_dim = config.hidden_size
        self.need_bilstm = need_bilstm
        if need_bilstm:
            self.bilstm = nn.LSTM(config.hidden_size, rnn_dim, num_layers=1, bidirectional=True, batch_first=True)
            self.out_dim = 2*rnn_dim
        self.liner = nn.Linear(self.out_dim, config.num_labels)
        self.crf = CRF(config.num_labels,batch_first=True)

        self.init_weights()


    def forward(self,input_ids=None,attention_mask=None,token_type_ids=None,labels=None,):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        if self.need_bilstm:
            sequence_output,_ = self.bilstm(sequence_output)
        sequence_output = self.dropout(sequence_output)
        sequence_output = self.liner(sequence_output)
        loss = -1 * self.crf(sequence_output, labels, mask=attention_mask.byte()) if labels != None else None
        output = self.crf.decode(sequence_output, attention_mask.byte())

        return [loss,output] if loss is not None else output