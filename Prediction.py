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
from Training import evaluate_model_seq_labeling


#load best/ saved model
load_model = "./output_dir/model.pt"
state_dict = torch.load(open(load_model, 'rb'))
model.load_state_dict(state_dict)
print("Loaded saved model")

model.to(device)

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
