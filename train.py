from DeepOD.deepod.models.time_series.timesnet import TimesNet, TimesNetModel
from DeepOD.deepod.models.time_series.anomalytransformer import AnomalyTransformer, AnomalyTransformerModel, fastinference
from DeepOD.deepod.models.time_series.dcdetector import DCdetector, DCdetectorModel
# from DeepOD.deepod.models.time_series import DIF
# from DeepOD.deepod.models.time_series import TcnED
# from DeepOD.deepod.models.time_series import NCAD
from DeepOD.deepod.models.time_series.tranad import TranAD, TranADNet
from DeepOD.deepod.models.time_series import COUTA
from DeepOD.deepod.models.time_series import USAD
# from DeepOD.deepod.models.time_series import PReNet
import torch
import torch.nn as nn
import torch.nn.functional as F
from test_data.model.LSTM_model import AAE_KTDATA, Encoder, TimeDistributed, Decoder, RecurrentAutoencoder, AutoencoderDataset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import copy
import pandas as pd
import numpy as np
from logger import timer
import gc
import mmap
def decision_function(seq_len,batch_size,X,net):

    seqs = get_sub_seqs(X, seq_len=seq_len, stride=1)
    dataloader = DataLoader(seqs, batch_size=batch_size,
                            shuffle=False, drop_last=False)

    net.eval()
    loss, _ = inference(seq_len,dataloader,net)  # (n,d)
    # print("loss",loss)
    loss_final = np.mean(loss, axis=1)  # (n,)

    padding_list = np.zeros([X.shape[0] - loss.shape[0], loss.shape[1]])
    loss_pad = np.concatenate([padding_list, loss], axis=0)
    loss_final_pad = np.hstack([0 * np.ones(X.shape[0] - loss_final.shape[0]), loss_final])
    return loss_final_pad
    
def inference(seq_len, dataloader,net):
    
    criterion = nn.MSELoss(reduction='none')
    temperature =100
    attens_energy = []
    preds = []

    for input_data in dataloader:  # test_set
        input = input_data.float().to(device)
        output, series, prior, _ = net(input)

        loss = torch.mean(criterion(input, output), dim=-1)
        
        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            if u == 0:
                series_loss = my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               seq_len)).detach()) * temperature
                prior_loss = my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            seq_len)),
                    series[u].detach()) * temperature
            else:
                series_loss += my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               seq_len)).detach()) * temperature
                prior_loss += my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            seq_len)),
                    series[u].detach()) * temperature
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = metric * loss
        cri = cri.detach().cpu().numpy()
        attens_energy.append(cri)
        preds.append(output)

    attens_energy = np.concatenate(attens_energy, axis=0)  # anomaly scores
    test_energy = np.array(attens_energy)  # anomaly scores

    return test_energy, preds  # (n,d)

def temporalize(data,lookback, feature_list): ## port_df를 numpy input
    print("Temporalization with DataFrame to Numpy")
    input_x = data.values
    print(np.shape(input_x))
    print(feature_list)
    X = np.reshape(input_x, (-1,len([feature_list])))
    output_X = []
    # output_y = []
    for i in range(np.shape(X)[0]-lookback+1):
        t = []
        # p = []
        t.append(X[i:i+lookback,:])
        # p.append(y[i:i+lookback,:])
        output_X.append(t)
        # output_y.append(p)
    
    return input_shaping(np.array(output_X),lookback,len([feature_list])) #, np.array(output_y)

def reverse_temporalize(origin_y, n_features, lookback): #windowed -> flatten data -> de windowed 
    output_X = []
    output_X = np.append(output_X, origin_y[0][:,:])
    for i in range(np.shape(origin_y)[0]-1):
        output_X= np.append(output_X,origin_y[i+1][-1,:])

    output_X = np.reshape(output_X,(-1, n_features))
    return output_X
    
    
def input_shaping(X_train, lookback,n_features): # x is Datarame
    X_train = X_train.reshape(X_train.shape[0], lookback, n_features)
    return X_train

class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)
    
        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]


##**model_configs,auto_hyper=True, model_type = model, data = prep ,col=col device="cpu",CV=True, epochs=300,seq_len =30,feature_list=['DAY_OF_WEEK','TIME_OF_DAY',col]
### Model Option =  MCD_LSTM, DCdetector, TimesNet, AnomalyTransformer, TranAD
@timer
def TrainModel(model_configs, model_path,model_type, data, col, device="cpu", CV = True, n_splits=5,epochs=20,seq_len =30, feature_list=['DAY_OF_WEEK','TIME_OF_DAY'],fine_tuning =False,partial_scaling=False):  
    
    device = device
    
    prep = copy.deepcopy(data)
    prep.port_df = prep.port_df.reset_index(drop=True)
    
    
    n_features = len(feature_list) 
    if len(col) == len(feature_list ):
        n_features = 1
    print("training",n_features)
    model_configs['seq_len'] = seq_len
    model_configs['n_features'] = n_features + 1 if partial_scaling == True else n_features
    model_configs['device'] = device
    model_configs['epochs_cv'] = np.floor(model_configs['epochs']/n_splits).astype(int)
    
    
    print("model architecture",model_type,model_configs)

    
    if model_type == 'MCD_LSTM':        
        tscv = BlockingTimeSeriesSplit(n_splits=n_splits)  # Train Set 과 Test Set을 제공 시계열 K-fold set 으로
        print("Test 데이터 세트를 준비합니다.")
        batch_size = model_configs['batch_size']
        lr         = model_configs['lr']
        lookback   = model_configs['seq_len']
        embedding_dim = model_configs['rep_dim']
        hidden_dim = model_configs['hidden_dim']
        pred_len = model_configs["pred_len"]
        # print((prep.port_df.shape))
             
        print(model_configs)
        model =RecurrentAutoencoder(seq_len, n_features=n_features, embedding_dim=embedding_dim)
        # model = Bi_LSTM(seq_len, n_features=n_features, embedding_dim=embedding_dim)
        model.to(device)
        best_mse = 100 #10.14
        criterion = nn.MSELoss().to(device) # seting 필요
        best_model_wts = copy.deepcopy(model.state_dict())
    
   
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        CV_fold_num = 1
        early_stop_counter = 0
        early_stopping_epochs = 10
        epochs = model_configs["epochs"]
        total_st = time.time() ## 전체 학습 소요 시간

        train_losses = []
        test_losses = []
        val_losses=[]
        hist =[]
             
            
        for train_index, test_index in tscv.split(prep.port_df):
            train, test = prep.port_df.loc[train_index,feature_list], prep.port_df.loc[test_index,feature_list]    
            
            train = temporalize(train,lookback = lookback, feature_list= feature_list)
            
            train = AutoencoderDataset(train)
            train_x = DataLoader(train[:-pred_len], batch_size=batch_size, shuffle=False)
            train_y = DataLoader(train[pred_len:], batch_size=batch_size, shuffle=False)
        
            test = temporalize(test, lookback = lookback, feature_list= feature_list)
            test = AutoencoderDataset(test)
            test_x = DataLoader(test[:-pred_len], batch_size=batch_size, shuffle=False)
            test_y = DataLoader(test[pred_len:], batch_size=batch_size, shuffle=False)
            #        Demodulation
            cv_st = time.time()  ## cv당 학습 소요 시간
            
            model = model.train()
            ##### Early Stoppping Parameter Setting
            # for epoch in tqdm(range(1, epochs+1), unit="epoch"):  
            for epoch in range(1, epochs+1):  
                for batch_x,batch_y in zip(train_x,train_y):
                
                    optimizer.zero_grad()
                    batch_x_tensor = batch_x.to(device) # seting 필요
                    batch_y_tensor = batch_y.to(device)
                    seq_pred = model.forward(batch_x_tensor)
                    
                    loss = criterion(seq_pred, batch_y_tensor)
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())
                # print(epoch)
                # model.eval() <- to implement MC-dropout
                ######### 여기 수정 하ㅏ자 
                
                with torch.no_grad():
                    for batch_x, batch_y in zip(test_x,test_y):
                        ## uncertainty 를 위해서 batch 반복할 필요가 있다. 최적의 반복법을 수행
                        test_x_tensor = batch_x.to(device) # seting 필요
                        test_y_tensor = batch_y.to(device) # seting 필요
                        seq_pred = model(test_x_tensor)
                        loss = criterion(seq_pred, test_y_tensor)
                        test_losses.append(loss.item())
                
                train_loss = np.mean(train_losses)
                val_loss =  np.mean(test_losses)
                
                if val_loss < best_mse:
                    best_mse = val_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print("best model is saved")
                
                cv_time2= time.time()
                tmp_hist = f'Epoch {(CV_fold_num-1)*epochs+(epoch)} : train loss:{train_loss}, val loss:{val_loss}'
                print(tmp_hist)
                # print(hist)
                hist.append([tmp_hist])
  
                if val_loss > best_mse:
                    early_stop_counter += 1
                    print(early_stop_counter)
                else:
                    early_stop_counter = 0
                    print("early_stop_counter is reset")
                
                if early_stop_counter >= early_stopping_epochs: 
                    print("Early Stopping")
                    break
            if early_stop_counter >= early_stopping_epochs: #epoch   

                print(f"Early Stopping, Time : {time.time()-total_st} ")
                
                break
            print(f"CV Fold Number {CV_fold_num}, CV_Time : {time.time()- cv_st}")
            print(tmp_hist)            
            CV_fold_num += 1

        return best_model_wts, hist
    
    elif model_type == 'DCdetector':

        
        tscv = BlockingTimeSeriesSplit(n_splits=n_splits)  # Train Set 과 Test Set을 제공 시계열 K-fold set 으로
        print("Train 세트를 준비합니다.")
        print(np.shape(prep.port_df))

        
        args = {}
        args["seq_len"] = model_configs["seq_len"]
        args["d_model"]  = model_configs["d_model"]
        # args["enc_in"]  = prep.port_df.shape[1]
        # args["c_out"]  = prep.port_df.shape[1]
        args["e_layers"] = model_configs["e_layers"]
        args["patch_size"]= model_configs["patch_size"]
        args["epochs"]= model_configs["epochs_cv"]
        # args["batch_size"]= model_configs["batch_size"]
        args["device"]   = model_configs['device']

        # args["enc_in"]   = model_configs["n_features"]
        # args["c_out"]    = model_configs["n_features"]
        # args["channel"]  = model_configs["n_features"]

        model = DCdetector(**args)

        CV_fold_num = 1

        total_st = time.time() ## 전체 학습 소요 시간
            
        for train_index, test_index in tscv.split(prep.port_df):
            train, test = prep.port_df.loc[train_index,feature_list], prep.port_df.loc[test_index,feature_list]

            train = np.reshape(train.values, (-1,n_features))  

            ##### Early Stoppping Parameter Setting
            if CV_fold_num == 1 :
                model.fit(train)
            elif CV_fold_num != 1 :
                model.fit_cv(train,cv_num=CV_fold_num)
            # score=model.decision_function(test.values)
            CV_fold_num += 1
        
        print("DCdetector Training Time :",time.time()-total_st,"초")
        return model.model.state_dict()

    
    elif model_type == 'TimesNet':
        tscv = BlockingTimeSeriesSplit(n_splits=n_splits)  # Train Set 과 Test Set을 제공 시계열 K-fold set 으로
        print("Test 데이터 세트를 준비합니다.")
        print(np.shape(prep.port_df))
        args = {}
        args["epochs"]= model_configs["epochs_cv"]
        args["seq_len"] = model_configs["seq_len"]
        args["pred_len"]   = model_configs["pred_len"]
        args["e_layers"] = model_configs["e_layers"]
        args["d_ff"] = model_configs["d_ff"]
        args["d_model"]= model_configs["d_model"]
        args["dropout"]  = model_configs["dropout"]
        args["top_k"]   = model_configs['top_k']
        args["num_kernels"]   = model_configs['num_kernels']
        # args["enc_in"]    = model_configs["n_features"]
        # args["c_out"]  = model_configs["n_features"]        
        model = TimesNet(**args)

        CV_fold_num = 1

        total_st = time.time() ## 전체 학습 소요 시간

        for train_index, test_index in tscv.split(prep.port_df):
            train, test = prep.port_df.loc[train_index,feature_list], prep.port_df.loc[test_index,feature_list]

            train = np.reshape(train.values, (-1,n_features))  

            if CV_fold_num == 1 :
                model.fit(train)
            elif CV_fold_num != 1 :
                model.fit_cv(train,cv_num=CV_fold_num)

            CV_fold_num += 1
            
        print("TimesNet Training Time :",time.time()-total_st,"초")
        return model.net.state_dict()

    
    elif model_type == 'AnomalyTransformer':
        tscv = BlockingTimeSeriesSplit(n_splits=n_splits)  # Train Set 과 Test Set을 제공 시계열 K-fold set 으로
        print("Train 데이터 세트를 준비합니다.")
        print(np.shape(prep.port_df))
        print(fine_tuning)

        if fine_tuning :
            args = {}
            args["win_size"] = model_configs["seq_len"]
            args["enc_in"]   = model_configs["n_features"] 
            args["c_out"]    = model_configs["n_features"] 
            args["e_layers"] = model_configs["e_layers"]
            args["device"]   = model_configs['device']
            
            model = AnomalyTransformerModel(**args).to(device)
            
            try:
                if device == 'cpu':
                    state_dict = torch.load(model_path, map_location='cpu')
                    if isinstance(state_dict, tuple):
                        state_dict = state_dict[0]
                        state_dict = dict(state_dict)
                    model.load_state_dict(state_dict )
                elif device != 'cpu':
                    state_dict = torch.load(model_path)
                    if isinstance(state_dict, tuple):
                        state_dict = state_dict[0]
                        state_dict = dict(state_dict)
                    model.load_state_dict(state_dict )
                print("Fine_tuning Start")
                
            except Exception as e:
                print(e)
                print("모델 없음")
                fine_tuning = False
                print("fine_tuning : ",fine_tuning)
                
                
        args = {}
        args["epochs"] = model_configs["epochs"]
        args["lr"]   =  model_configs["lr"]
        args["seq_len"] = model_configs["seq_len"]
        args["batch_size"]  = 16
        args["device"]   = model_configs['device']

        model_trainer = AnomalyTransformer(**args)

        if fine_tuning:
            print("Pre_trained model is set")
            model_trainer.net = model

        CV_fold_num = 1

        total_st = time.time() ## 전체 학습 소요 시간

            
        for train_index, test_index in tscv.split(prep.port_df):
            train, test = prep.port_df.loc[train_index,feature_list], prep.port_df.loc[test_index,feature_list]

            train = np.reshape(train.values, (-1,n_features))

            if fine_tuning:
                if CV_fold_num == 1 :
                    model_trainer.loss = np.inf
                    model_trainer.patient = 6
                    model_trainer.fit_cv(train,cv_num=CV_fold_num,partial_scaling=partial_scaling) #cv 메소드는 모델을 기존에서 구축한 놈을 활용함
                elif CV_fold_num != 1 :
                    model_trainer.fit_cv(train,cv_num=CV_fold_num,partial_scaling=partial_scaling)

                CV_fold_num += 1
            else:    
                if CV_fold_num == 1 :
                    model_trainer.fit(train,partial_scaling=partial_scaling)
                elif CV_fold_num != 1 :
                    model_trainer.fit_cv(train,cv_num=CV_fold_num,partial_scaling=partial_scaling)
                # score=model.decision_function(test.values)
                CV_fold_num += 1
            
        print("AnomalyTransformer Training Time :",time.time()-total_st,"초")
        return model_trainer.net.state_dict(), model_trainer.threshold_   
    
    
    elif model_type == 'TranAD':
        tscv = BlockingTimeSeriesSplit(n_splits=n_splits)  # Train Set 과 Test Set을 제공 시계열 K-fold set 으로
        print("Train 데이터 세트를 준비합니다.")
        print(np.shape(prep.port_df))
        args = {}
        args["seq_len"] = model_configs["seq_len"]
        args["epochs"]= model_configs["epochs_cv"]
        args["device"]   = model_configs['device']
        args["feats"]    = model_configs["n_features"]
        model = TranAD(**args)

        CV_fold_num = 1

        total_st = time.time() ## 전체 학습 소요 시간


        for train_index, test_index in tscv.split(prep.port_df):
            train, test = prep.port_df.loc[train_index,feature_list], prep.port_df.loc[test_index,feature_list]

            train = np.reshape(train.values, (-1,n_features))  

            if CV_fold_num == 1 :
                model.fit(train)
            elif CV_fold_num != 1 :
                model.fit_cv(train,cv_num=CV_fold_num)
            # score=model.decision_function(test.values)
            CV_fold_num += 1

        print("TranAD Training Time :",time.time()-total_st,"초")
        return model.model.state_dict()

@timer
def TestModel(model_configs, model_type, data, col, model_path='./', device="cpu", CV = True, epochs=20,seq_len =30,n_splits =5, feature_list=['DAY_OF_WEEK','TIME_OF_DAY'],partial_scaling=False):  
    device = device
    prep = copy.deepcopy(data)
    prep.port_df = prep.port_df.reset_index(drop=True)

    n_features = len(feature_list) 
    if len(col) == len(feature_list ):
        n_features = 1
    print("testing", n_features)
    model_configs['seq_len'] = seq_len
    model_configs['n_features'] = n_features + 1 if partial_scaling == True else n_features
    model_configs['device'] = device
    # model_configs['epochs'] = np.floor(model_configs['epochs']/n_split).astype(int)

    
    if model_type == 'MCD_LSTM':        
        embedding_dim = model_configs['rep_dim']
        pred_len = model_configs["pred_len"]
        tscv = BlockingTimeSeriesSplit(n_splits=n_splits)  # Train Set 과 Test Set을 제공 시계열 K-fold set 으로
        print("Test 데이터 세트를 준비합니다.")
        print(prep.port_df.shape)
        print(prep.port_df.head())
        batch_size = 1
        # lr = 0.0001
        lookback = model_configs['seq_len']
        
             

        model = RecurrentAutoencoder(lookback, n_features, embedding_dim = embedding_dim).to(device)
        # best_mse = 100#1
        criterion = nn.MSELoss(reduce = False, reduction='none').to(device) # seting 필요
        
        #### 폴더 생성         
        ##################### 모델 경로 수정 ##
        if device == 'cpu':
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, tuple):
                state_dict = state_dict[0]
                state_dict = dict(state_dict)
            model.load_state_dict(state_dict )
        elif device != 'cpu':
            state_dict = torch.load(model_path)
            if isinstance(state_dict, tuple):
                state_dict = state_dict[0]
                state_dict = dict(state_dict)
            model.load_state_dict(state_dict )


        
        CV_fold_num = 1
        
        total_st = time.time() ## 전체 학습 소요 시간

        test_losses = []
        val_losses=[]
        hist =[]
        dick = {}

        test = prep.port_df[col]
        test = temporalize(test, lookback = lookback, feature_list= feature_list)

        test_index=int(np.floor(np.shape(test)[0]*0.7))
        test = test[test_index:,:,:]

        test = AutoencoderDataset(test)

        test_x = DataLoader(test[:-pred_len], batch_size=batch_size, shuffle=False)
        test_y = DataLoader(test[pred_len:], batch_size=batch_size, shuffle=False)
        #        Demodulation
        cv_st = time.time()  ## cv당 학습 소요 시간

        ##### Early Stoppping Parameter SettingW
        rep_losses = []
        test_losses = []
        
        model.train()
        rep_losses = []
        with torch.no_grad():
            for i in range(100):
                test_losses = []
                for batch_x, batch_y in zip(test_x,test_y):
                    ## uncertainty 를 위해서 batch 반복할 필요가 있다. 최적의 반복법을 수행
                    test_x_tensor = batch_x.to(device) # seting 필요
                    test_y_tensor = batch_y.to(device) # seting 필요
                    seq_pred = model(test_x_tensor)
                    loss = criterion(seq_pred, test_y_tensor)
                    
                    test_losses.extend(np.mean(loss.tolist(),axis=1))

                # print(seq_pred, batch_y)
                rep_losses.append(test_losses)


        # print("np.shape",np.shape(test_losses))
        # print("np.shape",np.shape(rep_losses))
        # print("score shape:",np.shape(np.var(rep_losses,axis =0)))
        # print("len test:",len(test))
        # print("len date", len(prep.port_df['DATETIME'].values[test_index:][:-pred_len]) )

        result_dict = {}
        result_dict['DATETIME']    = prep.port_df['DATETIME'].values[test_index:][:np.shape(test_losses)[0]]
        result_dict['SCORE']       = np.mean(rep_losses,axis =0).reshape(-1,)
        result_dict['UNCERTAINTY'] = np.std(rep_losses,axis =0).reshape (-1,) 

        
        cv_time2= time.time()


        print(f"CV Fold Number {CV_fold_num}, CV_Time : {time.time()- cv_st}")


        ###저장할 Loss기록 = Hist 

        return  dick, result_dict                                             
    
    elif model_type == 'AnomalyTransformer':
        
        # tscv = BlockingTimeSeriesSplit(n_splits=n_splits)  # Train Set 과 Test Set을 제공 시계열 K-fold set 으로
        print("Test 데이터 세트를 준비합니다.")
#         print(prep.port_df.head())
        loading_st = time.time()        
        args = {}
        args["win_size"] = model_configs["seq_len"]
        args["enc_in"]   = model_configs["n_features"]
        args["c_out"]    = model_configs["n_features"]
        args["e_layers"] = model_configs["e_layers"]
        args["device"]   = model_configs['device']
        # args["n_heads"]   = 10

        model = AnomalyTransformerModel(**args).to(device)
        
        state_dict = torch.load(model_path, map_location='cpu')

        # model.load_state_dict(dict(state_dict[0]) if isinstance(state_dict, tuple) else  state_dict  )
        model.load_state_dict(dict(state_dict[0]))
        
        print(f"{model_type} loading Time :",time.time()-loading_st,"초")
        
        model.eval()


        inserting_st = time.time() 
        
        model = AnomalyTransformerModel(**args).to(device)
        
        args = {}
        args["epochs"] = model_configs["epochs"]
        args["lr"]   = model_configs["lr"]
        # args["c_out"]    = model_configs["n_features"]
        # args["e_layers"] = model_configs["e_layers"]
        args["seq_len"] = model_configs["seq_len"]
        args["batch_size"]   = 8
        args["device"]   = model_configs['device']

        model_testor = AnomalyTransformer(**args)
        model_testor.net = model
        
        print(f"{model_type} inserting Time :",time.time()-inserting_st,"초")
        
        # CV_fold_num = 1

        total_st = time.time() ## 전체 학습 소요 시간
        
        dick = {}        

        
        test_dict = prep.port_df.to_dict(orient = 'list') ## 이후 cv 기능을 적용을 위해 전체 db 남겨두기
        # print(test_dict)
        # print(n_features)
        # print(feature_list)
        test = np.reshape(test_dict[feature_list[0]], (-1,n_features))
        
        print(np.shape(test))
        
        loss =model_testor.decision_function(test)
        loss_dict = { "SCORE" : loss}
        
        result_dict = {}
        result_dict['DATETIME'] = test_dict['DATETIME']
        result_dict['SCORE'] = loss_dict['SCORE']
 
        print(f"{model_type} Testing Time :",time.time()-total_st,"초")

        del model_testor 
        del model
        del state_dict
        
        return dick, result_dict

def model_load(model_path):
    with open(model_path, "r+b") as f:
        mmaped_model = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        state_dict = torch.load(mmaped_model, map_location='cpu')
    return state_dict

@timer
def TestADModel(model, model_configs, model_type, data, col, model_path='./', device="cpu", CV = True, epochs=20,seq_len =30,n_splits =5, feature_list=['DAY_OF_WEEK','TIME_OF_DAY'],partial_scaling=False):  
#     torch.set_num_threads(1)
    
    device = device
    prep = copy.deepcopy(data)
    # model = copy.deepcopy(model)
    prep.port_df = prep.port_df.reset_index(drop=True)
    
    n_features = len(feature_list) 
    if len(col) == len(feature_list ):
        n_features = 1
    model_configs['seq_len'] = seq_len
    model_configs['n_features'] = n_features 
    model_configs['device'] = device
    # model_configs['epochs'] = np.floor(model_configs['epochs']/n_split).astype(int)


    print("Test 데이터 세트를 준비합니다.")
#         print(prep.port_df.head())
    loading_st = time.time()        
    args = {}
    args["win_size"] = model_configs["seq_len"]
    args["enc_in"]   = model_configs["n_features"]
    args["c_out"]    = model_configs["n_features"]
    args["e_layers"] = model_configs["e_layers"]
    args["device"]   = model_configs['device']
    # args["n_heads"]   = 10

#     model = AnomalyTransformer:Model(**args).to(device)

#     state_dict = torch.load(model_path, map_location='cpu')
    state_dict = model_load(model_path)
#     print(state_dict)
#     print(model)
    print(f"{model_type} loading Time :",time.time()-loading_st,"초")

    # model.load_state_dict(dict(state_dict[0]) if isinstance(state_dict, tuple) else  state_dict  )
    
    model.load_state_dict(dict(state_dict[0]))

    
    print(f"{model_type} State_inserting Time :",time.time()-loading_st,"초")
    
    model.eval()

    inserting_st = time.time() 
    
    dick = {}        

    test_dict = prep.port_df.to_dict(orient = 'list') ## 이후 cv 기능을 적용을 위해 전체 db 남겨두기
    test = np.reshape(test_dict[feature_list[0]], (-1,n_features))
    
    # print(np.shape(test))
    
    loss =    fastinference(model, test, device, seq_len=args["win_size"],)
    # del model, state_dict
   
    loss_dict = { "SCORE" : loss}
    print(f"{model_type} inferencing Time :",time.time()-inserting_st,"초")
    
    result_dict = {}
    result_dict['DATETIME'] = test_dict['DATETIME']
    result_dict['SCORE'] = loss_dict['SCORE']
    del model,state_dict
    # print(f"{model_type} Testing Time :",time.time()-infer_st,"초")
    return dick, result_dict

    ###############DataFrame Style
    # test = prep.port_df[feature_list].values ## 이후 cv 기능을 적용을 위해 전체 db 남겨두기
    # test = np.reshape(test, (-1,n_features))  
    # loss =model_testor.decision_function(test)

    # test_df = pd.DataFrame()
    # test_df['DATETIME'] = prep.port_df['DATETIME']
    # test_df['SCORE'] = loss
    # # print(f"{model_type} Testing Time :",time.time()-infer_st,"초")
    # return dick, test_df[args["seq_len"]-1:]


    
####Vaild
@timer
def ValidModel(model_configs, model_type, data, col, threshold,model_path='./', device="cpu", CV = True, epochs=20,seq_len =30,n_splits =5, feature_list=['DAY_OF_WEEK','TIME_OF_DAY'],partial_scaling=False):  
    device = device
    prep = copy.deepcopy(data)
    prep.port_df = prep.port_df.reset_index(drop=True)
    
    n_features = len(feature_list) 
    if len(col) == len(feature_list ):
        n_features = 1
    model_configs['seq_len'] = seq_len
    model_configs['n_features'] = n_features + 1 if partial_scaling == True else n_features
    model_configs['device'] = device
    # model_configs['epochs'] = np.floor(model_configs['epochs']/n_split).astype(int)

    
    if model_type == 'MCD_LSTM':        
        embedding_dim = model_configs['rep_dim']
        pred_len = model_configs["pred_len"]
        tscv = BlockingTimeSeriesSplit(n_splits=n_splits)  # Train Set 과 Test Set을 제공 시계열 K-fold set 으로
        print("Test 데이터 세트를 준비합니다.")
        print(prep.port_df.shape)
        print(prep.port_df.head())
        batch_size = 128
        # lr = 0.0001
        lookback = model_configs['seq_len']
        
             

        model = RecurrentAutoencoder(lookback, n_features, embedding_dim = embedding_dim).to(device)
        # best_mse = 100#1
        criterion = nn.MSELoss(reduce = False, reduction='none').to(device) # seting 필요
        
        #### 폴더 생성         
        ##################### 모델 경로 수정 ##
        if device == 'cpu':
            state_dict = torch.load(model_path, map_location='cpu')
            if isinstance(state_dict, tuple):
                state_dict = state_dict[0]
                state_dict = dict(state_dict)
            model.load_state_dict(state_dict )
        elif device != 'cpu':
            state_dict = torch.load(model_path)
            if isinstance(state_dict, tuple):
                state_dict = state_dict[0]
                state_dict = dict(state_dict)
            model.load_state_dict(state_dict )


        
        CV_fold_num = 1
        
        total_st = time.time() ## 전체 학습 소요 시간

        test_losses = []
        val_losses=[]
        hist =[]
        dick = {}

        test = prep.port_df[col]
        test = temporalize(test, lookback = lookback, feature_list= feature_list)

        test_index=int(np.floor(np.shape(test)[0]*0.7))
        test = test[test_index:,:,:]

        test = AutoencoderDataset(test)

        test_x = DataLoader(test[:-pred_len], batch_size=batch_size, shuffle=False)
        test_y = DataLoader(test[pred_len:], batch_size=batch_size, shuffle=False)
        #        Demodulation
        cv_st = time.time()  ## cv당 학습 소요 시간

        ##### Early Stoppping Parameter SettingW
        rep_losses = []
        test_losses = []
        
        model.train()
        rep_losses = []
        pred_y_set = []
        with torch.no_grad():
            
            
            for i in range(100):
                pred_y = []
                test_losses = []
                for batch_x, batch_y in zip(test_x,test_y):
                    ## uncertainty 를 위해서 batch 반복할 필요가 있다. 최적의 반복법을 수행
                    test_x_tensor = batch_x.to(device) # seting 필요
                    test_y_tensor = batch_y.to(device) # seting 필요
                    seq_pred = model(test_x_tensor)
                    loss = criterion(seq_pred, test_y_tensor)
                    pred_y.extend(seq_pred.tolist())
                    print("결과물")
                    print( np.shape( seq_pred.tolist() )  )
                    print( np.shape( pred_y)  )
                    test_losses.extend(np.mean(loss.tolist(),axis=1))
                    print("loss")
                    print( np.shape( test_losses)  )

                # print(seq_pred, batch_y)
                pred_y = np.squeeze(pred_y)
                pred_y_set.append(pred_y)
                rep_losses.append(test_losses)
                print("loss_with")
                print( np.shape( pred_y_set)  )
                print( np.shape( rep_losses)  )

            
        cv_time2= time.time()


        print(f"CV Fold Number {CV_fold_num}, CV_Time : {time.time()- cv_st}")


        ###저장할 Loss기록 = Hist 

        return  pred_y_set, test_losses, rep_losses                                             
 
