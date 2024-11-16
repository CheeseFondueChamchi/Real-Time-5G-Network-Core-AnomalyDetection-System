#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import warnings

warnings.filterwarnings("ignore")
import sys
sys.path.append('..')
import os
import glob
import pickle

# from database.logger import logger, timed, formatter
from collections import Counter
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#####################################
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from pickle import dump
# from time_series_func import temporalize, scale, input_shaping
torch.manual_seed(1)

class AAE_KTDATA(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self,item):
        sample = self.data[item]
        return sample
class Encoder(nn.Module):
    def __init__(self, seq_len=10, n_features=4, embedding_dim=64):
        super(Encoder, self).__init__()
        self.seq_len, self.n_features = seq_len, n_features
        self.embedding_dim , self.hidden_dim = embedding_dim, embedding_dim
        self.rnn1 = nn.LSTM(\
                            input_size = n_features \
                            , hidden_size = self.embedding_dim \
                            , num_layers=1 \
                            , batch_first=True \
                            , bidirectional = False\
                            # ,dropout =0.7
                           )
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        x, (hidden_n,_) = self.rnn1(x)
        x = self.dropout(x)
        return x[:,-1,:self.embedding_dim]
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first = False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)
        if self.batch_first:
            y= y.contiguous().view(x.size(0),-1, y.size(-1)) # contiguous 연속적인 메모리를 반환하는 메서드 / 
        else:
            y= y.view(-1, x.size(1), y.size(-1))
        return y
class Decoder(nn.Module):
    def __init__(self, seq_len=10, embedding_dim = 64, n_features=4):
        super(Decoder,self).__init__()
        self.seq_len, self.embedding_dim = seq_len, embedding_dim
        self.hidden_dim, self.n_features = embedding_dim, n_features
        self.rnn1 = nn.LSTM(\
                            input_size = self.embedding_dim  \
                            , hidden_size=self.hidden_dim \
                            , num_layers=1 \
                            , batch_first =True \
                            , bidirectional = False\
                            # ,dropout =0.7
                           )
        self.dropout = nn.Dropout(0.3)
        self.output_layer = torch.nn.Linear(self.hidden_dim, n_features)
        self.timedist = TimeDistributed(self.output_layer, batch_first=True)
        
    def forward(self, x):
        x= x.reshape(-1,1,self.embedding_dim).repeat(1,self.seq_len,1)
        x = self.dropout(x)
        x, (hidden_n, cell_n) = self.rnn1(x)
        x = self.dropout(x)
        return self.timedist(x[:,:,:])#
    
class RecurrentAutoencoder(nn.Module):
    def __init__(self, seq_len, n_features, embedding_dim = 64):
        super(RecurrentAutoencoder, self).__init__()
        self.encoder = Encoder(seq_len, n_features, embedding_dim)
        self.decoder = Decoder(seq_len, embedding_dim, n_features)
    def forward(self, x):
        x = self.encoder(x)

        x = self.decoder(x)

        return x

# class Bi_LSTM(nn.Module):
#     def __init__(self,seq_len, n_features, embedding_dim=64, device='cuda', batch_size=32):
#         super(Bi_LSTM, self).__init__()
#         self.lstm = nn.LSTM(seq_len, embedding_dim,
#                             num_layers=1, bidirectional=True,batch_first=True)
#         self.output_layer = nn.Linear(embedding_dim, n_features)

#     def forward(x):

#         x, (hidden_n, cell_n) = self.lstm(x)
#         return self.output_layer(x)
    
    
class block(nn.Module):
    def __init__(self, in_channel, out_channels):
        super(block, self).__init__()
        bn = nn.BatchNorm1d(in_channel)

    def forward(self,x):
        out = bn(x)  #[B, N, T] -> [B, N, T]

class AutoencoderDataset(Dataset):
    def __init__(self,x):
        self.x=x
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        data = torch.FloatTensor(self.x[idx,:,:])
        return data
# @timed
def trainer(model_conf,df, ru, logger):
    st = time.time()

    df.sort_values(by='datetime', ascending = True, inplace=True)
    df = df.replace(np.nan,0)
    model_conf = model_conf['MODEL']
    feature= [ 'rlculbyte', 'rlcdlbyte', 'airmaculbyte','airmacdlbyte']
    
    
    input_y = df.loc[:,feature].values
    test_len=int(input_y.shape[0]*0.7)
    input_y = np.reshape(input_y, (-1,len(feature)))
    n_features = input_y.shape[1]  # number of features

    lookback = 10 # Equivalent to 50 min of past data.
    print("date length :",input_y.shape[0])

    train_y=input_y[:test_len]
#    print(train_y.shape)
    test_y =input_y[test_len:]
#    print(test_y.shape)
    scaler = StandardScaler().fit(train_y)
    if not os.path.exists(model_conf['scale_savepath']):
        os.mkdir(model_conf['scale_savepath'])
    with open(os.path.join(model_conf['scale_savepath'],f'{ru}.pkl'), 'wb') as f:
        dump(scaler,f)
#     X_train_scaled = train_x
#     X_valid_scaled = test_x
    train_y = scaler.transform(train_y)
    test_y = scaler.transform(test_y)

    train_y, _ = temporalize(X = train_y, y = train_y, timesteps = lookback)
    test_y, _ = temporalize(X = test_y, y = test_y, timesteps = lookback)

    train_y = input_shaping(train_y, lookback,n_features)
    test_y = input_shaping(test_y, lookback,n_features)
#     train_y_0 = input_shaping(train_y, lookback,n_features)[:-1]
#     test_y_0 = input_shaping(test_y, lookback,n_features)[:-1]
#     train_y_1 = input_shaping(train_y, lookback,n_features)[1:]
#     test_y_1 = input_shaping(test_y, lookback,n_features)[1:]
    
    train_y = AutoencoderDataset(train_y)
    test_y = AutoencoderDataset(test_y)
#     train_y = AutoencoderDataset(train_y)
#     test_y = AutoencoderDataset(test_y)
#     train_y = AutoencoderDataset(train_y)
#     test_y = AutoencoderDataset(test_y)
    
    train_y = DataLoader(train_y, batch_size=32, shuffle=False)
    test_y = DataLoader(test_y, batch_size=32, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    epochs =100 # test
    batch = 32
    
    lr = 0.0001
    model = RecurrentAutoencoder(lookback, n_features, embedding_dim = 40)
    model.to(device) # seting 필요
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_mse = 100#10.14
    criterion = nn.MSELoss().to(device) # seting 필요
    best_model_wts = copy.deepcopy(model.state_dict())

    if not os.path.exists(model_conf['lstm_ae_savepath']):
        os.mkdir(model_conf['lstm_ae_savepath'])
    
    for epoch in range(1, epochs+1):
        #Demodulation
        model = model.train()
        train_losses = []
        test_losses = []
        
        for batch_idx, batch_x in enumerate(train_y):
            optimizer.zero_grad()
            batch_x_tensor = batch_x.to(device) # seting 필요
            seq_pred = model.forward(batch_x_tensor)
            loss = criterion(seq_pred, batch_x_tensor)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        val_losses=[]
        model.eval()
        
        #Validation
        with torch.no_grad():
            for batch_idx, batch_x in enumerate(test_y):
                test_tensor = batch_x.to(device) # seting 필요
                seq_pred = model(test_tensor)
                loss = criterion(seq_pred, test_tensor)
                test_losses.append(loss.item())
        train_loss = np.mean(train_losses)
        val_loss =  np.mean(test_losses)
        if val_loss < best_mse:
            best_mse = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            if not os.path.exists(model_conf['valid_loss_savepath']):
                os.mkdir(model_conf['valid_loss_savepath'])
            torch.save(model.state_dict(),os.path.join(model_conf['lstm_ae_savepath'],f'{ru}.ckpt'))
            with open(os.path.join(model_conf['valid_loss_savepath'],f'{ru}.pkl'), 'wb') as f:
                dump(test_losses,f)
                print ("save the best valid_losses")

# print(f'Save the Best State for {ru} at', os.path.join(model_conf['lstm_ae_savepath'],f'{ru}.ckpt'))

        if epoch %20 == 0 :
            print(f'Epoch {epoch}: train loss:{train_loss}, val loss:{val_loss}')
    ed = time.time()
    logger.info(f'insert sql executed time: {ed-st:.2f}')
        
    
    
    
    
    
    
    
    
    
    
#     #lstm_autoencoder = LSTM_autoencoder_bi_one(X_train,'tanh',100)

    
    
    
#     for step in range(total_step):
#         if (step
    
    
#     cp = ModelCheckpoint(filepath=os.path.join(model_conf['lstm_ae_savepath'],f"{ru}.h5"),
#                                 save_best_only=True, save_weights_only=False, monitor = 'val_loss', mod = 'min',
#                                 verbose=True)
#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=False, patience=30)
#     # lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: lr = 1e-4 * 10*(20 / epoch)) 


#     lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=False)
#     for i in range(0,3):
#         lstm_autoencoder_history = lstm_autoencoder.fit(X_train, X_train, 
#                                                         epochs=epochs, 
#                                                         batch_size=batch, 
#                                                         validation_data=(X_valid, X_valid),
#                                                         verbose=True,callbacks=[cp,es])
#         lstm_autoencoder_history = lstm_autoencoder.fit(X_train[:-1], X_train[1:], 
#                                                     epochs=int(epochs/5), 
#                                                     batch_size=batch, 
#                                                     validation_data=(X_valid, X_valid),
#                                                     verbose=True,callbacks=[cp,es])
#         lstm_autoencoder_history = lstm_autoencoder.fit(X_train, X_train, 
#                                                     epochs=int(epochs/5), 
#                                                     batch_size=batch, 
#                                                     validation_data=(X_valid, X_valid),
#                                                     verbose=True,callbacks=[cp,es])

