#Evaluatung the NER on Arabic dialect dataset

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



#load best/ saved model
load_model = "./output_dir/model.pt"
state_dict = torch.load(open(load_model, 'rb'))
model.load_state_dict(state_dict)
print("Loaded saved model")

model.to(device)


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


predict_file = "./low-resource-seq-labeling/data/UNLABELED/unlabeled_aoc/sents.unlabeled"
pred_examples = data_processor.get_pred_examples(predict_file)
pred_features = data_processor.convert_examples_to_features(pred_examples, label_list, 320, model.encode_word)

pred_data = create_ner_dataset(pred_features)
f1_score, report, y_true, y_pred = evaluate_model_seq_labeling(model, pred_data, label_list, eval_batch_size, use_crf, device, pred=True)

print("\n%s", report)
output_pred_file = "./content/"
with open(output_pred_file, "w") as writer:
    for ex, pred in zip(pred_examples, y_pred):
        writer.write("Ex text: {}\n".format(ex.text))
        writer.write("Ex labels: {}\n".format(ex.label))
        writer.write("Ex preds: {}\n".format(pred))
        writer.write("*******************************\n")
