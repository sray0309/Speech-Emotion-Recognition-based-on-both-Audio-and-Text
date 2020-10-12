import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim

data = pd.read_csv("sentence_label_comb.csv").dropna().reset_index(drop=True)
sentence_length = []
for i in range(len(data)):
    length = len(data.sentence[i].split())
    sentence_length.append(length)

sentence = []
label = []
for i in range(len(data)):
    sentence.append(data.sentence[i])
    label.append(data.label[i])

train_sentence = sentence[0: int(len(data)*0.9)]
train_label = label[0: int(len(data)*0.9)]

test_sentence = sentence[int(len(data)*0.9):]
test_label = label[int(len(data)*0.9):]
vocab_list = []
for i in sentence:
    for j in i.split():
        if j not in vocab_list:
            vocab_list.append(j)
tokenizer = Tokenizer(num_words=2700, oov_token='<UNK>')
tokenizer.fit_on_texts(train_sentence)
word_index = tokenizer.word_index
train_sequence = tokenizer.texts_to_sequences(train_sentence)
train_sequence = pad_sequences(train_sequence, maxlen = 35, padding='post', truncating='post')

test_sequence = tokenizer.texts_to_sequences(test_sentence)
test_sequence = pad_sequences(test_sequence, maxlen = 35, padding='post', truncating='post')

class SentimentRNN(nn.Module):

    def __init__(self):
        super(SentimentRNN, self).__init__()
        vocab_size = 2700
        output_size = 4
        embedding_dim = 64
        hidden_dim = 100
        drop_prob=0.2
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim,
                            dropout = drop_prob, bidirectional = True, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(
            # This layer enables user to write conv layer and fc layer in one nn.Sequential model
            nn.Flatten(start_dim = 1, end_dim = -1),
            nn.Linear(in_features = 3500*2, out_features = 4),
            nn.Softmax(dim =1)
        )

    def forward(self, x):
        x = x.long()
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        return out
        
model = SentimentRNN()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
train_tensor = Variable(torch.tensor(train_sequence),requires_grad = False)
train_label_tensor = Variable(torch.tensor(train_label),requires_grad = False)

test_tensor = Variable(torch.tensor(test_sequence),requires_grad = False)
test_label_tensor = Variable(torch.tensor(test_label),requires_grad = False)
for itera in range(1000):
    print("iter: {:2d}".format(itera))
    outputs = model(train_tensor)
    optimizer.zero_grad()
    loss = loss_function(outputs, train_label_tensor)
    loss.backward()
    optimizer.step()
    _, outputs_label = outputs.max(dim=1)

    accuracy = int(sum(outputs_label == train_label_tensor))/len(train_label_tensor)
    print("train accuray: {:.2f}, loss: {:.2e}".format(accuracy, loss))
    outputs = model(test_tensor)
    _, y_pred = outputs.max(dim=1)
    loss_test = loss_function(outputs, test_label_tensor)
    accuracy = int(sum(y_pred == test_label_tensor))/len(test_label_tensor)
    print("test accuray: {:.2f}, loss: {:.2e}".format(accuracy, loss_test))
