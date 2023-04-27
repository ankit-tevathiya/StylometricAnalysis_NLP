import csv
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from argparse import ArgumentParser

import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn

parser = ArgumentParser()

parser.add_argument("-g", "--gold", dest="gold", help="path to gold labels csv")
parser.add_argument("-p", "--pred", dest="pred", help="path to predicted labels csv")

args = parser.parse_args()

gold_stream = open(args.gold, "r")
gold_labels_str = gold_stream.readlines()
gold_stream.close()
gold_labels_str = [x.strip() for x in gold_labels_str[1:]]
gold_labels_01 = np.array([1 if x == "same" else 0 for x in gold_labels_str])

pred_stream = open(args.pred, "r")
pred_labels_str = pred_stream.readlines()
pred_stream.close()
pred_labels_str = [x.strip() for x in pred_labels_str[1:]]
pred_labels_01 = np.array([1 if x == "same" else 0 for x in pred_labels_str])

print("Accuracy:", accuracy_score(gold_labels_01, pred_labels_01))
print("F1 score:", f1_score(gold_labels_01, pred_labels_01))
