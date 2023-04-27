import pandas as pd

from argparse import ArgumentParser

import warnings


def warn(*args, **kwargs):
  pass

warnings.warn = warn

parser = ArgumentParser()

parser.add_argument("-s", "--set", dest="option",
                    help="give the name of the set you want to predict on")

args = parser.parse_args()

if args.option and args.option not in ["train", "dev", "test"]:
    train_df = pd.read_csv("/data/X_test.csv")
elif args.option and args.option in ["train", "dev", "test"]:
    train_df = pd.read_csv("/data/X_"+args.option+".csv")
else:
    train_df = pd.read_csv("/data/X_test.csv")

a = len(train_df)
output = ["same"]*a

stream = open(args.option+"_pred.csv", "w")
for i in output[:10]:
  stream.write(i+"\n")
stream.close()
