from transformers import AdamW, get_cosine_schedule_with_warmup

from Ner.bert_bilstm_crf import config


def optimizer(model):
    if config.full_fine_tuning:
        # model.named_parameters(): [bert, bilstm, classifier, crf]
        bert_optimizer = list(model.bert.named_parameters())
        #lstm_optimizer = list(model.bilstm.named_parameters())
        classifier_optimizer = list(model.liner.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            #{'params': [p for n, p in lstm_optimizer if not any(nd in n for nd in no_decay)],
             #'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
            #{'params': [p for n, p in lstm_optimizer if any(nd in n for nd in no_decay)],
             #'lr': config.learning_rate * 5, 'weight_decay': 0.0},
            {'params': [p for n, p in classifier_optimizer if not any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': config.weight_decay},
            {'params': [p for n, p in classifier_optimizer if any(nd in n for nd in no_decay)],
             'lr': config.learning_rate * 5, 'weight_decay': 0.0},
            {'params': model.crf.parameters(), 'lr': config.learning_rate * 5}
        ]
    # only fine-tune the head classifier
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer]}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, correct_bias=False)
    train_steps_per_epoch = config.train_size // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=(config.epoch_num // 10) * train_steps_per_epoch,
                                                num_training_steps=config.epoch_num * train_steps_per_epoch)
    return optimizer,scheduler
