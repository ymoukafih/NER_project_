# -*- coding: utf-8 -*-
"""Simplified NER.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1i8XnwnrqnJudmmnWPgp79KIEGfA7dvBY
"""

from __future__ import absolute_import, division, print_function

import argparse
import csv
import json
import logging
import os
import random
import sys
import sklearn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_transformers import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from seqeval.metrics import classification_report, f1_score, accuracy_score
from tqdm import tqdm, trange
from fairseq.models.roberta import XLMRModel
from TorchCRF import CRF



class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, label_id, input_mask=None, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask

class SequenceLabelingProcessor:
    """Processor for the CoNLL-2003 data set."""
    def __init__(self, task):
        assert task in ['ner', 'pos']
        if task == 'ner':
            self.labels = ["O", "B-PERS", "I-PERS", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "U"]

        elif task =='pos':
            self.labels = ['TB', 'WB', 'PART', 'V', 'ADJ', 'DET', 'HASH', 'NOUN', 'PUNC',
                           'CONJ', 'PREP', 'PRON', 'EOS', 'CASE', 'EMOT', 'NSUFF', 'NUM',
                                  'URL', 'ADV', 'MENTION', 'FUT_PART', 'ABBREV', 'FOREIGN', 'PROG_PART', 'NEG_PART','U']

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "train.txt")), "train")

    def get_pred_examples(self, file_dir):
        """See base class."""
        return self._create_examples(self._read_file(file_dir),"pred")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "valid.txt")), "valid")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "test.txt")), "test")

    def get_unlabeled_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_file(os.path.join(data_dir, "sents.unlabeled")), "unlabeled")


    def get_labels(self):
        return self.labels

    def _read_file(self, filename):
        '''
        read file
        '''
        f = open(filename, encoding='utf-8', errors='ignore')
        data = []
        sentence = []
        label = []

        # get all labels in file

        for i, line in enumerate(f, 1):
            if not line.strip() or len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n" or line[0] == '.' or line.split()[0]=='EOS':
                if len(sentence) > 0:
                    data.append((sentence, label))
                    sentence = []
                    label = []
                continue

            splits = line.split()
            if len(splits) <= 1:
                logging.info("skipping line")
                continue
            assert len(splits) >= 2, "error on line {}. Found {} splits".format(i, len(splits))
            word, tag = splits[0], splits[-1]

            if tag not in self.get_labels():
                logging.info("ignoring unknown tag {} in line {}".format(tag, i))
                continue
            #if tag in ['WB', "TB"]:
            #    tag = "IGNORE"

            sentence.append(word.strip())
            label.append(tag.strip())

        if len(sentence) > 0:
            data.append((sentence, label))
            print(label)
            sentence = []
            label = []
        return data

    def _create_examples(self, lines, set_type):
        examples = []

        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))

        logging.info("max sentence length = %d" %(max(len(ex.text_a.split()) for ex in examples)))
        return examples


    def convert_examples_to_features(self, examples, label_list, max_seq_length, encode_method):
        """Converts a set of examples into XLMR compatible format

        * Labels are only assigned to the positions correspoinding to the first BPE token of each word.
        * Other positions are labeled with 0 ("IGNORE")

        """
        ignored_label = "IGNORE"
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        label_map[ignored_label] = 0  # 0 label is to be ignored

        features = []
        for (ex_index, example) in enumerate(examples):

            textlist = example.text_a.split(' ')
            labellist = example.label
            labels = []
            valid = []
            label_mask = []
            token_ids = []

            for i, word in enumerate(textlist):
                tokens = encode_method(word.strip())  # word token ids
                token_ids.extend(tokens)  # all sentence token ids
                label_1 = labellist[i]
                for m in range(len(tokens)):

                    if m == 0:  # only label the first BPE token of each work
                        labels.append(label_1)
                        valid.append(1)
                        label_mask.append(1)
                    else:
                        labels.append(ignored_label)  # unlabeled BPE token
                        label_mask.append(0)
                        valid.append(0)

            if len(token_ids) >= max_seq_length - 1:  # trim extra tokens
                token_ids = token_ids[0:(max_seq_length-2)]
                labels = labels[0:(max_seq_length-2)]
                valid = valid[0:(max_seq_length-2)]
                label_mask = label_mask[0:(max_seq_length-2)]

            # adding <s>
            token_ids.insert(0, 0)
            labels.insert(0, ignored_label)
            label_mask.insert(0, 0)
            valid.insert(0, 0)

            # adding </s>
            token_ids.append(2)
            labels.append(ignored_label)
            label_mask.append(0)
            valid.append(0)

            assert len(token_ids) == len(labels)
            assert len(valid) == len(labels)

            label_ids = []
            for i, _ in enumerate(token_ids):
                label_ids.append(label_map[labels[i]])

            assert len(token_ids) == len(label_ids)
            assert len(valid) == len(label_ids)

            input_mask = [1] * len(token_ids)

            while len(token_ids) < max_seq_length:
                token_ids.append(1)  # token padding idx
                input_mask.append(0)
                label_ids.append(label_map[ignored_label])  # label ignore idx
                valid.append(0)
                label_mask.append(0)

            while len(label_ids) < max_seq_length:
                label_ids.append(label_map[ignored_label])
                label_mask.append(0)

            assert len(token_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(valid) == max_seq_length
            assert len(label_mask) == max_seq_length

            features.append(
                InputFeatures(input_ids=token_ids,
                              input_mask=input_mask,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
        return features

def create_ner_dataset(features):

    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    all_valid_ids = torch.tensor(
        [f.valid_ids for f in features], dtype=torch.long)
    all_lmask_ids = torch.tensor(
        [f.label_mask for f in features], dtype=torch.uint8)

    return TensorDataset(
        all_input_ids, all_label_ids, all_lmask_ids, all_valid_ids)

def create_clf_dataset(features):

    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)

    return TensorDataset(
        all_input_ids, all_label_ids)

def evaluate_model_seq_labeling(model, eval_dataset, label_list, batch_size, use_crf, device, pred=False):
     # Run prediction for full data
     eval_sampler = SequentialSampler(eval_dataset)
     eval_dataloader = DataLoader(
          eval_dataset, sampler=eval_sampler, batch_size=batch_size)

     model.eval() # turn of dropout

     y_true = []
     y_pred = []

     label_map = {i: label for i, label in enumerate(label_list, 1)}
     label_map[0] = "IGNORE"


     for input_ids, label_ids, l_mask, valid_ids in eval_dataloader:

          input_ids = input_ids.to(device)
          label_ids = label_ids.to(device)

          valid_ids = valid_ids.to(device)
          l_mask = l_mask.to(device)

          with torch.no_grad():
               logits = model(input_ids, labels=None, labels_mask=None,
                              valid_mask=valid_ids)

          if use_crf:
               predicted_labels = model.decode_logits(logits, mask=l_mask, device=device)
          else :
               predicted_labels = torch.argmax(logits, dim=2)

          predicted_labels = predicted_labels.detach().cpu().numpy()
          label_ids = label_ids.cpu().numpy()

          for i, cur_label in enumerate(label_ids):
               temp_1 = []
               temp_2 = []

               for j, m in enumerate(cur_label):
                   if valid_ids[i][j] and label_map[m] not in ['WB' , 'TB']: #'PROG_PART', 'NEG_PART']:  # if it's a valid label
                         temp_1.append(label_map[m])
                         temp_2.append(label_map[predicted_labels[i][j]])

               assert len(temp_1) == len(temp_2)
               y_true.append(temp_1)
               y_pred.append(temp_2)

     report = classification_report(y_true, y_pred, digits=4)
     f1 = f1_score(y_true, y_pred, average='macro')
     acc = accuracy_score(y_true, y_pred)

     s = "Accuracy = {}".format(acc)
     print(s)
     report +='\n\n'+ s

     if 'NOUN' in label_map.values():
         print("Returning acc")
         f1=acc

     if pred:
         return f1, report, y_true, y_pred
     return f1, report

#!pip install fairseq
#!pip install TorchCRF

class XLMRForTokenClassification(nn.Module):

    def __init__(self, pretrained_path, n_labels, hidden_size, dropout_p, label_ignore_idx=0,
                 head_init_range=0.04, device='cuda'):
        super().__init__()

        self.n_labels = n_labels

        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.classification_head = nn.Linear(hidden_size, n_labels)

        self.label_ignore_idx = label_ignore_idx

        self.xlmr = XLMRModel.from_pretrained(pretrained_path)
        self.model = self.xlmr.model
        self.dropout = nn.Dropout(dropout_p)

        self.device = device

        # initializing classification head
        self.classification_head.weight.data.normal_(
            mean=0.0, std=head_init_range)


    def forward_generator(self, inputs_ids):
        '''
        Computes a forward pass to generate embeddings

        Args:
            inputs_ids: tensor of shape (bsz, max_seq_len), pad_idx=1
            labels: temspr pf soze (bsz)

        '''
        transformer_out, _ = self.model(inputs_ids, features_only=True)
        generator_representation = transformer_out.mean(dim=1) # bsz x hidden
        return generator_representation


    def forward(self, inputs_ids, labels, labels_mask, valid_mask, get_sent_repr= False):
        '''
        Computes a forward pass through the sequence agging model.
        Args:
            inputs_ids: tensor of size (bsz, max_seq_len). padding idx = 1
            labels: tensor of size (bsz, max_seq_len)
            labels_mask and valid_mask: indicate where loss gradients should be propagated and where
            labels should be ignored

        Returns :
            logits: unnormalized model outputs.
            loss: Cross Entropy loss between labels and logits

        '''
        transformer_out, _ = self.model(inputs_ids, features_only=True)
        sent_repr = transformer_out.mean(dim=1)

        out_1 = F.relu(self.linear_1(transformer_out))
        out_1 = self.dropout(out_1)
        logits = self.classification_head(out_1)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.label_ignore_idx)
            # Only keep active parts of the loss
            if labels_mask is not None:
                active_loss = valid_mask.view(-1) == 1
                active_logits = logits.view(-1, self.n_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
                #print("Preds = ", active_logits.argmax(dim=-1))
                #print("Labels = ", active_labels)
            else:
                loss = loss_fct(
                    logits.view(-1, self.n_labels), labels.view(-1))

            if get_sent_repr:
                return loss, sent_repr
            return loss
        else:
            return logits

    def encode_word(self, s):
        """
        takes a string and returns a list of token ids
        """
        tensor_ids = self.xlmr.encode(s)
        # remove <s> and </s> ids
        return tensor_ids.cpu().numpy().tolist()[1:-1]

#!pip install pytorch_transformers

#hyper-parameters

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

gradient_accumulation_steps = 1
train_batch_size = 16
max_seq_length = 320
num_train_epochs = 5
learning_rate = 0.00001
adam_epsilon = 1e-8
eval_batch_size, use_crf = 128, False
train_batch_size = train_batch_size // gradient_accumulation_steps

hidden_size = 768
pretrained_path = "./pretrained_models/xlmr.base"
output_dir = "./output_dir"
data_dir = "./low-resource-seq-labeling/data/NER/ANERCorp"
device = 'cuda' if (torch.cuda.is_available()) else 'cpu'

weight_decay = 0.01
warmup_proportion = 0.1
fp16 = False
max_grad_norm = 1.0

self_training = False
no_pbar = False

if gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            gradient_accumulation_steps))

os.makedirs("output_dir")

#data preparation
do_train = True

data_processor = SequenceLabelingProcessor(task='ner')
label_list = data_processor.get_labels()
num_labels = len(label_list) + 1  # add one for IGNORE label

train_examples = None
num_train_optimization_steps = 0

if do_train:
    train_examples = data_processor.get_train_examples(data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / train_batch_size / gradient_accumulation_steps) * num_train_epochs

model_cls = XLMRForTokenClassification

model = model_cls(pretrained_path=pretrained_path,
                                       n_labels=num_labels, hidden_size=hidden_size,
                                       dropout_p=0.3, device=device)

model.to(device)

no_decay = ['bias', 'final_layer_norm.weight']
params = list(model.named_parameters())

optimizer_grouped_parameters = [
    {'params': [p for n, p in params if not any(
        nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in params if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
warmup_steps = int(warmup_proportion * num_train_optimization_steps)

label_map = {i: label for i, label in enumerate(label_list, 1)}

if do_train:
    train_features = data_processor.convert_examples_to_features(
        train_examples, label_list, max_seq_length, model.encode_word)

    print("***** Running training *****")
    print("  Num examples = ", len(train_examples))
    print("  Batch size = ", train_batch_size)
    print("  Num steps = ", num_train_optimization_steps)

    train_data = create_ner_dataset(train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
    train_data, sampler=train_sampler, batch_size=train_batch_size)


    val_examples = data_processor.get_dev_examples("./low-resource-seq-labeling/data/NER/ANERCorp")
    val_features = data_processor.convert_examples_to_features(
        val_examples, label_list, max_seq_length, model.encode_word)

    val_data = create_ner_dataset(val_features)
    best_val_f1 = 0.0

    n_iter=0
    optimizer = AdamW(optimizer_grouped_parameters,
                          lr=learning_rate, eps=adam_epsilon)
    scheduler = WarmupLinearSchedule(
            optimizer, warmup_steps=warmup_steps, t_total=num_train_optimization_steps)
    patience = 0

    while 1:
      n_iter+=1
      print(len(train_dataloader))
      loss_fct = nn.BCELoss()
      for epoch_ in tqdm(range(num_train_epochs), desc="Epoch", disable=no_pbar):
        tr_loss = 0
        tbar = tqdm(train_dataloader, desc="Iteration", disable=no_pbar)
        model.train()
        for step, batch in enumerate(tbar):
          batch = tuple(t.to(device) for t in batch)
          input_ids, label_ids, l_mask, valid_ids, = batch
          loss, _ = model(input_ids, label_ids, l_mask, valid_ids, get_sent_repr=True)
          tr_loss += loss.item()
          optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

          if (step + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()

          tbar.set_description('Loss = %.4f' %(tr_loss / (step+1)))

        print("Evaluating on validation set...\n")

        f1, report = evaluate_model_seq_labeling(model, val_data, label_list, eval_batch_size, use_crf, device)
        if f1 > best_val_f1:
            best_val_f1 = f1
            print("\nFound better f1=%.4f on validation set. Saving model\n" %(f1))
            print("\n%s\n" %(report))

            torch.save(model.state_dict(), open(os.path.join(output_dir, 'model.pt'), 'wb'))
            patience=0

        else :
            print("\nNo better F1 score: {}\n".format(f1))
            patience+=1

      if patience >= 10:
        print("No more patience. Existing")
        break

      for g in optimizer.param_groups:
        g['lr'] = learning_rate

      scheduler.step(0)
