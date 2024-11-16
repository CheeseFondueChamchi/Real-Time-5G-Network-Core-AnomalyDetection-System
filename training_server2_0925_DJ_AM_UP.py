import pymysql
import yaml
import argparse
import getpass
from preprossesor import Preporcessor, dataload, train_folder_dir, CreatePath,seed_everything
from train import TestModel,TrainModel
import os
import pandas as pd
import numpy as np
import torch
from postprosseor import AnomalyGraphCapture 
from z_score import ZscoreCalc
import matplotlib.pyplot as plt
import pickle
import csv
import sys
import datetime
from multiprocessing import Pool
from DataManager import PathCreator,LoadThreshold,SaveThreshold,Thresholding, RulePathCreator ,GetData, GetLastTime,GetTrainData,GetInferenceData,InferenceDirCreator, TrainDirCreator,GetTrainDataByDG,GetTrainData_R
import gc
import time

def train(args, train_list):
    seq_len = args['seq_len']
    n_split = args['CV']
    device =  args['GPU']
    lastmin = 60*24*60
    i =0
    for db,table,port,col,r_feature in train_list:
        print(i)
        i += 1 
        try: 
            seed_everything(42)
            model_list = ["MCD_LSTM", "DCdetector", "TimesNet","TranAD","AnomalyTransformer","Rule_Model"]
            model_type = model_list[-2]

            table_path = table.replace("|","_")
            data_path  = os.path.join(db,table_path)
            port_path = port.replace("/","_")
            col_path=col.replace("/","_")
            feature_list = col
            print(r_feature)
            if r_feature != "A": # "A is normal feature"            
                train_df, _= GetTrainData_R(db=db,table=table,col=col,port=port,r_feature=r_feature,lastmin= lastmin)
            else:
                train_df, _= GetTrainData(db=db,table=table,col=col,port=port,lastmin= lastmin)
            print( train_df)
            print("train date time",train_df.shape)

            if train_df.shape[0] < 1000 :#: 학습시 nan의 주범
                print("없어용")
                print("DG 확인해볼께요 ~")

                train_df, _= GetTrainDataByDG(db=db,table=table,col=col,port=port,lastmin= lastmin)
                if train_df.shape[0] < 1000 :#: 학습시 nan의 주범
                    print("없어용")
                    print("지나가요~~~~~~")
                    continue

            lastdate = train_df["DATETIME"].max()
            startdate = train_df["DATETIME"].min()            
            print(startdate,lastdate)

            model_path, scaler_path ,z_path, df_path, Th_path, save_path_pkl,save_path_csv = PathCreator(args,data_path,model_type,port_path,col_path,lastdate,db,table_path)
            if not (args["FT"]) :
                print("Fine Tuning is on False")
                if os.path.exists(model_path) and os. path.exists(scaler_path) and os.path.exists(Th_path):
                    print("있어용")
                    print("지나가요~~~~~~")
                    continue                    

            if lastdate is None :#:

                print("시간이 없어용","time:",lastdate)
                print("지나가용")
                continue
            prep = Preporcessor(train_df)

            yaml_path = './test_data/model/configs/configs.yaml'


            model_list = ["MCD_LSTM", "DCdetector", "TimesNet","TranAD","AnomalyTransformer","Rule_Model"]
            model_list = [model_list[-2]]

            table_path = table.replace("|","_")
            data_path  = os.path.join(db,table_path)


            with open(yaml_path) as f:
                d = yaml.safe_load(f)
                try:
                    model_configs = d[model_type] #args.model
                except KeyError:
                    print(f'config file does not contain default parameter settings of {model_type} ') #{args.model}
                    model_configs = {}

            #### 저장 경로
            TrainDirCreator(args, data_path, model_list,lastdate)


            prep.PortSelection(port,startdate=startdate, lastdate=lastdate, lasttime=lastmin, padding =True) ### PADDING 제거 
            # prep.DayToNum() 

            port_path = port.replace("/","_")
            col_path=col.replace("/","_")


            model_path, scaler_path ,z_path, df_path, Th_path, save_path_pkl,save_path_csv = PathCreator(args,data_path,model_type,port_path,col_path,lastdate,db,table_path)

            feature_list = ["R_" + col if r_feature != "A" else col]
            prep.Scaler( col=feature_list, scaler_type = "Standard", path= scaler_path, save=True)
            # prep.Diff(col=col, n_diff=1)


    ############# 모델 불러오기 및 재학습 ###########
            model_w=TrainModel(model_configs = model_configs,
                model_type = model_type, 
                n_splits=n_split,
                data = prep ,
                col=col, 
                device= device, #"ms",
                CV=True, 
                epochs=30,
                seq_len =seq_len,
                feature_list=feature_list,
                fine_tuning =args["FT"],
                model_path = model_path,
                partial_scaling=False)

            torch.save(model_w, model_path)
            # print(torch.load(model_path))
            print("Save Model-weights in :",model_path)

            _, test_df = TestModel(model_configs = model_configs, 
                model_type = model_type, 
                model_path = model_path,
                n_splits=n_split,
                data = prep ,col=col, 
                device='cpu',
                CV=True, 
                epochs=10,
                seq_len =seq_len,
                feature_list=feature_list,
                partial_scaling=False)    


            x =test_df["SCORE"][args["seq_len"]-1:] #.values
            date = test_df["DATETIME"]
            print("testset Score :",x)
            threshold = SaveThreshold(date,x,col,0.02,Th_path,cho = True)
            print(" Threshold :", threshold)
            print("Save TH in :",Th_path)
            threshold = LoadThreshold(Th_path)
        
        except Exception as e:
            print(e)
            while True:
                try:
                                        # MySQL 서버에 연결 시도
                    if "CSCF" in db:
                        connection = pymysql.connect(host='172.21.223.201', user='ai5gcore', password='ai5gcore1234!', database=f'{db}', port=33062)
                        print("Connected to MySQL server")
                    else:
                        connection = pymysql.connect(host='172.21.223.201', user='ai5gcore', password='ai5gcore1234!', database=f'{db}', port=33060)
                        print("Connected to MySQL server")

                    # 연결이 성공했으므로 반복문을 종료
                    connection.close()
                    break
                except Exception as e:
                    # 연결 실패한 경우
                    print(f"Failed to connect to MySQL server: {e}")

                    # 30분 대기
                    time.sleep(180)

            # print(E)
            # continue
        finally:
            gc.collect()
            torch.cuda.empty_cache()

# ######### Save Training Result at Local Directory########################
        # prep.InverseScaler(col, path=scaler_path)
        # test_df["VALUE"]=prep.port_df[f"{col}_ORIGIN"]
        # test_df[f"{col}'s Abnormality"] = False
        # test_df[f"{col}'s Threshold"] = threshold
        # test_df.loc[ test_df["SCORE"] >= threshold,f"{col}'s Abnormality"] = True    
        # test_df.to_csv(df_path,index=False)
        # AnomalyGraphCapture(test_df[6912:],col,"SCORE",df_path,seq_len,"AS",threshold)
        print("Save trained Score in", df_path)  
#################################################
def process_train(args, train_item):
    db, table, port, col = train_item
    # train 함수 호출
    train(args, [(db, table, port, col)])
        
if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description = "5G Core Predictive-Prevention")
    parser.add_argument("--server",type=str, default='server1') #server1 or server2
    parser.add_argument("--mp_processor",type=int, default=1)
    parser.add_argument("--model_path",type=str,default='/mnt/lv03')
    parser.add_argument("--checker_path",type=str,default="./trainable_checker_v6.1.csv")
    parser.add_argument("--save_path",type=str,default="/mnt/lv03/save_path")
    parser.add_argument("--GPU",type=str, default='cuda:0')
    parser.add_argument("--DB_set", type=int, default=1) # 1: [0:8] , 2: [8:16] ,3 : [16:24], 4 : [24:32]
    parser.add_argument("--FT", action=argparse.BooleanOptionalAction, default=False)    
    parser.add_argument("--seq_len",type=int, default=8)
    parser.add_argument("--CV",type=int, default=1)
    args = parser.parse_args()
    args = vars(args)
    
    trainable_csv=pd.read_csv("./trainable_checker_v6.1.csv").drop_duplicates()
    trainable_csv = trainable_csv.loc[trainable_csv['AT_model'] ==True,:]
    trainable_csv_tmp1 = trainable_csv.set_index(["DB_name", "table_name", "port", "col_name","R_feature"]).sort_values(by=["DB_name", "table_name", "port"])

    nf = { "SET_1": ['AMF22_5M','SMF02_5M','SMF03_5M','UPF01_5M','UPF03_5M','UPF03_5M','AMF21_5M','AMF22_5M','DJ_SMF01_5M','UPF01_5M','UPF01_5M'], ## 1차 학습
          "SET_2": ['AMF21_5M', 'UPF01_5M'], # 2차 학습
         }
                                 
    train_list = []
    
    for idx in trainable_csv_tmp1.index:
        if idx[0] in nf["SET_3"]:
            train_list.append(idx)
    
    train(args,train_list)
    
    print("Training is Completed")
                                  

