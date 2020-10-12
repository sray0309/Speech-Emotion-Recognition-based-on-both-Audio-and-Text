import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import time
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

# Load audio MFCC
MFCC = (np.load("FC_MFCC12EDA.npy")).transpose(0,2,1)  # Transpose
data_train = MFCC[0:int(MFCC.shape[0]*0.9)]
data_test = MFCC[int(MFCC.shape[0]*0.9):]
# Load label
file=open("FC_label.txt")
lines=file.readlines()
label = []
for line in lines:
    line = line.strip().split('\n')
    label.append(line)
label_int = (np.array(label)[:,0]).astype(int)
label_train = label_int[0:int(MFCC.shape[0]*0.9)]
label_test = label_int[int(MFCC.shape[0]*0.9):]
audio_data_train_tensor = Variable(torch.tensor(data_train),requires_grad = False)
audio_data_test_tensor = Variable(torch.tensor(data_test),requires_grad = False)
audio_label_train_tensor = Variable(torch.tensor(label_train),requires_grad = False)
audio_label_test_tensor = Variable(torch.tensor(label_test),requires_grad = False)

import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
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

tokenizer = Tokenizer(num_words=3000, oov_token='<UNK>')
tokenizer.fit_on_texts(train_sentence)
word_index = tokenizer.word_index

train_sequence = tokenizer.texts_to_sequences(train_sentence)
train_sequence = pad_sequences(train_sequence, maxlen = 35, padding='post', truncating='post')

test_sequence = tokenizer.texts_to_sequences(test_sentence)
test_sequence = pad_sequences(test_sequence, maxlen = 35, padding='post', truncating='post')

text_data_train_tensor = Variable(torch.tensor(train_sequence),requires_grad = False)
text_label_train_tensor = Variable(torch.tensor(train_label),requires_grad = False)
text_data_test_tensor = Variable(torch.tensor(test_sequence),requires_grad = False)
text_label_test_tensor = Variable(torch.tensor(test_label),requires_grad = False)
# Define network architecture
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        
        self.conv0 = nn.Sequential(nn.Conv1d(in_channels=39, out_channels=10, kernel_size = 4),nn.ReLU())
        self.lstm_audio = nn.LSTM(input_size=10, hidden_size=16, batch_first = True, bidirectional = True)
        self.fc0 = nn.Sequential(nn.Flatten(start_dim = 1, end_dim = -1),nn.Linear(in_features = 23904, out_features = 4))
        
        vocab_size = 3000
        embedding_dim = 64
        self.embedding1 = nn.Embedding(vocab_size, embedding_dim)
        self.lstm1 = nn.LSTM(input_size = embedding_dim, hidden_size = 100,
                            dropout = 0.2, bidirectional = True, batch_first=True)
        self.fc1 = nn.Sequential(
            nn.Flatten(start_dim = 1, end_dim = -1),
            nn.Linear(in_features = 3500*2, out_features = 4)
        )
        self.final = nn.Sequential(nn.Linear(in_features = 8, out_features = 4), nn.Softmax(dim =1))
        
    def forward(self,x,y):
        Conv_out0 = self.conv0(x)
        LSTM_out0,_ = self.lstm_audio(Conv_out0.transpose(1,2))
        FC_out0 = self.fc0(LSTM_out0)
        y = y.long()
        embeds1 = self.embedding1(y)
        lstm_out1, _ = self.lstm1(embeds1)
        FC_out1 = self.fc1(lstm_out1)
        final_input = torch.cat((FC_out0, FC_out1), dim=1)
        output = self.final(final_input)
        return output

# Generate distribution of dataset
ang = 0
hap = 0
sad = 0
neu = 0
for i in range(len(label)):
    if(label[i]==0):
        ang = ang+1
    if(label[i]==1):
        hap = hap+1
    if(label[i]==0):
        sad = sad+1
    if(label[i]==0):
        neu = neu+1
print('ang=',ang,'hap=',hap,'sad=',sad,'neu=',neu)
labels=['Angry (1103files)','Happy (1636files)','Sad (1103files)','Neutral (1103files)']
X=[ang,hap,sad,neu]
colors = ['yellow','lightcoral','dodgerblue','coral']
fig = plt.figure()
plt.pie(X,labels=labels,colors = colors,autopct='%1.2f%%')
plt.show()
plt.savefig("PieChart.jpg")

# Train and Test
Model = model()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Model.parameters(), lr=1e-3, weight_decay=1e-3)
total_loss = []
epoch = []

for itera in range(10000):
    print("iter: {:2d}".format(itera))
    outputs = Model(audio_data_train_tensor.float(),text_data_train_tensor)
    optimizer.zero_grad()
    loss = loss_function(outputs, text_label_train_tensor)
    loss.backward()
    optimizer.step()
    total_loss.append(loss)
    epoch.append(epoch)
    _, outputs_label = outputs.max(dim=1)
    accuracy = int(sum(outputs_label == audio_label_train_tensor))/len(text_label_train_tensor)
    print("train accuray: {:.2f}, loss: {:.2e}".format(accuracy, loss))
    outputs = Model(audio_data_test_tensor.float(),text_data_test_tensor)
    _, y_pred = outputs.max(dim=1)
    loss_test = loss_function(outputs, text_label_test_tensor)
    accuracy = int(sum(y_pred == audio_label_test_tensor))/len(text_label_test_tensor)
    print("test accuray: {:.2f}, loss: {:.2e}".format(accuracy, loss_test))
# Save the trained network
torch.save(Model.state_dict(), 'model_trained.pkl')
    
