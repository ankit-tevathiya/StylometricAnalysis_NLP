import torch            #
import torch.nn as nn

import numpy as np      #
import pandas as pd     #

import time
import math

from pymagnitude import *       #

from sklearn.linear_model import LogisticRegression     #
from sklearn.metrics import accuracy_score, f1_score

import warnings

warnings.filterwarnings("ignore")

data_path = "data/"

vectors_dir = Magnitude(data_path+"glove.6B.100d.magnitude")

embeddings = {}
for entry in vectors_dir:
    embeddings[entry[0]] = entry[1]


def remove_quotes(list_of_strings):
    return [strng.replace("'", "") for strng in list_of_strings]


def get_embedding(list_of_tokens):
    mtx = []
    for token in list_of_tokens:
        mtx.append(embeddings.get(token, embeddings.get("unk")))

    mtx = np.array(mtx)
    return mtx


def create_df(filepath, dataset):
    df = pd.read_csv(filepath + "X_" + dataset + ".csv", encoding="utf-8")
    df["label"] = pd.read_csv(filepath + "y_" + dataset + ".csv")

    # convert each column from a string representation of a list to an actual list
    df.text_a = df.text_a.apply(lambda x: remove_quotes(x[2:-2].split(", ")))
    df.text_b = df.text_b.apply(lambda x: remove_quotes(x[2:-2].split(", ")))

    # get embeddings for each text
    df["a_embedding"] = df.text_a.apply(lambda x: get_embedding(x))
    df["b_embedding"] = df.text_b.apply(lambda x: get_embedding(x))

    return df


class RNN(nn.Module):
    def __init__(self, input_size=100, hidden_size=150, n_layers=1, bi_dir=False):
        super(RNN, self).__init__()
        self.input_size = input_size  # number of features in an x (word embedding)
        self.hidden_size = hidden_size  # size of hidden state
        self.n_layers = n_layers  # number of GRU layers (stacked)

        self.gru = nn.GRU(input_size, hidden_size, n_layers,
                          bidirectional=bi_dir, batch_first=True)

    def forward(self, input_mtx):
        output, _ = self.gru(input_mtx)  # default initializes h0 as 0-vector for each element in the batch
        return output


def train(text_a_mtx, text_b_mtx, gold_labels):
    text_a_mtx = torch.tensor(text_a_mtx).float()
    text_b_mtx = torch.tensor(text_b_mtx).float()

    output_a = rnn_GRUa(text_a_mtx)  # calls forward method
    output_b = rnn_GRUb(text_b_mtx)  # output dims: (len_dataset, 100, 150 or 300)

    # average over all h_t's
    a_avg_mtx = torch.mean(output_a, 1, True)  # mtx dims: (len_dataset, 1, 150 or 300)
    b_avg_mtx = torch.mean(output_b, 1, True)

    # calculate cosine similarity
    cos = nn.CosineSimilarity(dim=2)
    c_s = cos(a_avg_mtx, b_avg_mtx)  # c_s dims: (len_dataset, 1) ~= column of dataframe

    # train logistic regression classifier on single feature
    clfr = LogisticRegression(random_state=0).fit(c_s.detach().numpy(), gold_labels)

    probas = clfr.predict_proba(c_s.detach().numpy())
    probas_tensor = torch.tensor(probas).float()
    gold_labels_01 = np.array([1 if x == "same" else 0 for x in gold_labels])
    gold_labels_tensor = torch.tensor(gold_labels_01).long()

    # calculate loss
    loss = criterion(probas_tensor, gold_labels_tensor)

    # backpropagate
    loss.requires_grad = True
    loss.backward()
    optimizer.step()

    return loss.data, c_s.detach().numpy(), clfr


def get_cosine_similarity(text_a_mtx, text_b_mtx):
    text_a_mtx = torch.tensor(text_a_mtx).float()
    text_b_mtx = torch.tensor(text_b_mtx).float()

    output_a = rnn_GRUa(text_a_mtx)  # passes through current states of RNNs
    output_b = rnn_GRUb(text_b_mtx)

    # average over all h_t's
    a_avg_mtx = torch.mean(output_a, 1, True)  # mtx dims: (len_dataset, 1, 150 or 300)
    b_avg_mtx = torch.mean(output_b, 1, True)

    # calculate cosine similarity
    cos = nn.CosineSimilarity(dim=2)
    c_s = cos(a_avg_mtx, b_avg_mtx)  # c_s dims: (len_dataset, 1) ~= column of dataframe

    return c_s.detach().numpy()


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


train_df = create_df(data_path, "train")

train_a_mtx = np.stack(train_df.a_embedding.to_numpy())
train_b_mtx = np.stack(train_df.b_embedding.to_numpy())

dev_df = create_df(data_path, "dev")
dev_a_mtx = np.stack(dev_df.a_embedding.to_numpy())
dev_b_mtx = np.stack(dev_df.b_embedding.to_numpy())

hidden_size = 150
n_layers = 1
lr = 0.001

rnn_GRUa = RNN()
rnn_GRUb = RNN()

params = list(rnn_GRUa.parameters()) + list(rnn_GRUb.parameters())
optimizer = torch.optim.Adam(params, lr=lr)

criterion = nn.NLLLoss()

start = time.time()
training_losses = []
dev_losses = []

n_epochs = 20
save_every = 1
epochs = range(1, n_epochs + 1)
epoch_clfr = None
for epoch in epochs:
    trn_loss, trn_cos_sim, epoch_clfr = train(train_a_mtx, train_b_mtx, train_df.label)

    if epoch % save_every == 0:
        training_losses.append(trn_loss)
        print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, trn_loss))

        dev_cos_sim = get_cosine_similarity(dev_a_mtx, dev_b_mtx)
        d_probas = epoch_clfr.predict_proba(dev_cos_sim)
        d_probas_tensor = torch.tensor(d_probas).float()
        d_gold_labels_01 = np.array([1 if x == "same" else 0 for x in dev_df.label])
        d_gold_labels_tensor = torch.tensor(d_gold_labels_01).long()

        # calculate loss
        dev_loss = criterion(d_probas_tensor, d_gold_labels_tensor)
        dev_losses.append(dev_loss.data)
        print("dev loss:", dev_loss.data)

        # calculate training accuracy, F1
        trn_pred_labels = epoch_clfr.predict(trn_cos_sim)
        print("training acc:", accuracy_score(train_df.label, trn_pred_labels))
        trn_pred_labels01 = np.array([1 if x == "same" else 0 for x in trn_pred_labels])
        trn_labels01 = np.array([1 if x == "same" else 0 for x in train_df.label])
        print("training f1:", f1_score(trn_labels01, trn_pred_labels01))

        # calculate dev accuracy, F1
        dev_pred_labels = epoch_clfr.predict(dev_cos_sim)
        print("dev acc:", accuracy_score(dev_df.label, dev_pred_labels))
        dev_pred_labels01 = np.array([1 if x == "same" else 0 for x in dev_pred_labels])
        print("dev f1:", f1_score(d_gold_labels_01, dev_pred_labels01))

        print()
