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
    
#train_sentence = sentence[0: int(len(data)*0.6)]
#train_label = label[0: int(len(data)*0.6)]

#val_sentence = sentence[int(len(data)*0.6):int(len(data)*0.8)]
#val_label = label[int(len(data)*0.6):int(len(data)*0.8)]

train_sentence = sentence[0: int(len(data)*0.9)]
train_label = label[0: int(len(data)*0.9)]

test_sentence = sentence[int(len(data)*0.9):]
test_label = label[int(len(data)*0.9):]

vocab_list = []
for i in sentence:
    for j in i.split():
        if j not in vocab_list:
            vocab_list.append(j)
# len(vocab_list)
# 3147
# since the vocabulary size of the sentence is 3147, so I would pick the 
# 3000 most common words
tokenizer = Tokenizer(num_words=3000, oov_token='<UNK>')
tokenizer.fit_on_texts(train_sentence)
word_index = tokenizer.word_index

train_sequence = tokenizer.texts_to_sequences(train_sentence)
train_sequence = pad_sequences(train_sequence, maxlen = 35, padding='post', truncating='post')

#val_sequence = tokenizer.texts_to_sequences(val_sentence)
#val_sequence = pad_sequences(val_sequence, maxlen = 35, padding='post', truncating='post')

test_sequence = tokenizer.texts_to_sequences(test_sentence)
test_sequence = pad_sequences(test_sequence, maxlen = 35, padding='post', truncating='post')

text_data_train_tensor = Variable(torch.tensor(train_sequence),requires_grad = False)
text_label_train_tensor = Variable(torch.tensor(train_label),requires_grad = False)
text_data_test_tensor = Variable(torch.tensor(test_sequence),requires_grad = False)
text_label_test_tensor = Variable(torch.tensor(test_label),requires_grad = False)

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
        self.dropout1 = nn.Dropout(0.3)
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
        drop_out1 = self.dropout1(lstm_out1)
        FC_out1 = self.fc1(drop_out1)
        final_input = torch.cat((FC_out0, FC_out1), dim=1)
        output = self.final(final_input)
        #output = nn.Linear(final_input, out_features=4)
        return output

Model_r = model()
Model_r.load_state_dict(torch.load('model_trained.pkl')) 
Model_r.eval()
outputs = Model_r(audio_data_test_tensor.float(),text_data_test_tensor)
_, y_pred = outputs.max(dim=1)
accuracy = int(sum(y_pred == audio_label_test_tensor))/len(text_label_test_tensor)
print("test accuray: {:.2f}".format(accuracy))
mat = np.zeros(shape=(4,4))
mat_new = np.zeros(shape=(4,4))
for i in range(0,len(text_label_test_tensor)):
    m = text_label_test_tensor[i]
    n = y_pred[i]
    mat[m,n] = mat[m,n] + 1

for i in range(4):
    mat[i,:] = mat[i,:]/sum(mat[i,:])
import seaborn as sns
import matplotlib as mpl
sns.set(style = "whitegrid",color_codes = True)
#np.random.seed(sum(map(ord,"categorical")))
#df_corr = someDataFrame.corr()
ax = sns.heatmap(mat, annot=True) #notation: "annot" not "annote"
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
#heatmap = sns.heatmap(mat,annot = True)
plt.show()
#accuracy = int(sum(y_pred == audio_label_test_tensor))/len(text_label_test_tensor)
#print("test accuray: {:.2f}".format(accuracy))
