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

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv1d(in_channels=39, out_channels=10, kernel_size = 4),nn.ReLU())
        self.lstm_audio = nn.LSTM(input_size=10, hidden_size=16, batch_first = True, bidirectional = True)
        self.fc0 = nn.Sequential(nn.Flatten(start_dim = 1, end_dim = -1),nn.Linear(in_features = 23904, out_features = 4))
        
    def forward(self,x):
        Conv_out0 = self.conv0(x)
        LSTM_out0,_ = self.lstm_audio(Conv_out0.transpose(1,2))
        FC_out0 = self.fc0(LSTM_out0)
        output = FC_out0
        #output = nn.Linear(final_input, out_features=4)
        return output

Model = model()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Model.parameters(), lr=1e-3, weight_decay=1e-3)
#optimizer = optim.SGD(list(Audio_model.parameters()) + list(Text_model.parameters()), lr=1e-3,weight_decay = 1e-3)
total_loss = []
epoch = []
# Train
for itera in range(10000):
    print("iter: {:2d}".format(itera))
    outputs = Model(audio_data_train_tensor.float())
    optimizer.zero_grad()
    loss = loss_function(outputs, audio_label_train_tensor)
    loss.backward()
    optimizer.step()
    total_loss.append(loss)
    epoch.append(epoch)
    _, outputs_label = outputs.max(dim=1)
    
    accuracy = int(sum(outputs_label == audio_label_train_tensor))/len(audio_label_train_tensor)
    print("train accuray: {:.2f}, loss: {:.2e}".format(accuracy, loss))
    outputs = Model(audio_data_test_tensor.float())
    _, y_pred = outputs.max(dim=1)
    loss_test = loss_function(outputs, audio_label_test_tensor)
    accuracy = int(sum(y_pred == audio_label_test_tensor))/len(audio_label_test_tensor)
    print("test accuray: {:.2f}, loss: {:.2e}".format(accuracy, loss_test))
# Save the model
torch.save(Model.state_dict(), 'model_trained_audio_only.pkl')
    

