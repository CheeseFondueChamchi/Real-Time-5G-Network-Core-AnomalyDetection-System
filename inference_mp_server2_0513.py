##### Distinct DB, Table, Port Group inference version  ****** No Column*******

import pymysql
import yaml
import argparse
import getpass
from preprossesor import Preporcessor, dataload, train_folder_dir, CreatePath, merge_dict_lists
from train import TestModel,TrainModel,TestADModel
import os
from itertools import groupby
from operator import itemgetter
import pandas as pd
import numpy as np
import torch
from postprosseor import AnomalyGraphCapture 
from z_score import ZscoreCalc
# import matplotlib.pyplot as plt
import pickle
import csv
import sys
import datetime
# from multiprocessing import Pools
import multiprocessing as mp
from DataManager import PathCreator,LoadThreshold,SaveThreshold,Thresholding, RulePathCreator ,GetData, GetLastTime,GetTrainData,InferenceDirCreator,InsertOuputData,GetInferenceTableData,GetInferenceData,GetInferenceData_R
import time 
# import cProfile
#####SQL 추가 
from logger import timer
import ast
import json
import gc
from DeepOD.deepod.models.time_series.anomalytransformer import  AnomalyTransformerModel
import time
from concurrent.futures import ProcessPoolExecutor
@timer
def inference(args, train_list):
    torch.set_num_threads(1)
    
    seq_len = args['seq_len']
    n_split = args['CV']
    device =  args['GPU']
    lastdate =args["last_time"] 
    lastmin = 40
    tmp_port = ""
    tmp_table = ""
    err_indexing = 0

    test_df_per_port = {"DATETIME" :[],"TABLE":[],"SYSTEM":[],"PORT":[],"COLUMNS":[],"VALUE":[],"SCORE":[],"DETECTION":[]} 
    
    tmp_table = ""
    ## 테이블 단위로 데이터 가져와보기 -> 소용 없음
    
    #    args["win_size"] = model_configs["seq_len"]
    model_args = {}
    model_args["win_size"] = seq_len
    model_args["enc_in"]   = 1
    model_args["c_out"]    = 1
    model_args["e_layers"] = 3
    model_args["device"]   = device

    model = AnomalyTransformerModel(**model_args).to(device)
    
    prev_table = None
    i = 0
    for db,table,port,col,r_feature in train_list:

#         try:
        print("r_feature",r_feature)
        if r_feature != "A": #tables = tables.loc[:,["DATETIME","PORT","SYSTEM",col,"R_"+col]]
#                 continue
            train_df,_ = GetInferenceData_R( db=db, lastdate=lastdate ,table=table,col=col,port=port,r_feature=r_feature,lastmin = 40)     
        else:
#                 continue
            train_df,_ = GetInferenceData( db=db, lastdate=lastdate ,table=table,col=col,port=port,lastmin = 40)     
        # print(train_df)
        model_list = ["MCD_LSTM", "DCdetector", "TimesNet","TranAD","AnomalyTransformer","Rule_Model"]
        model_type = model_list[-2]

        table_path = table.replace("|","_")
        data_path  = os.path.join(db,table_path)
        port_path = port.replace("/","_")
        col_path  = col.replace("/","_")

        feature_list = "R_" + col if r_feature != "A" else col
#         print("feature_list :",feature_list)


        model_path, scaler_path ,z_path, df_path, Th_path, save_path_pkl,save_path_csv = PathCreator(args,data_path,model_type,port_path,col_path,lastdate,db,table_path)
        print("model_path", model_path)
        if (not(os.path.exists(scaler_path)) or not(os.path.exists(model_path)) or not(os.path.exists(Th_path))):
            # do something
            print(os.path.exists(scaler_path))
            print(os.path.exists(model_path)) 
            print(os.path.exists(Th_path))
            print("Not Properly Saved")
            continue

        if len(train_df.loc[:,col].values) < 4:
            print("Not Enough Inference Data Length ")
            continue

        system_name = train_df["SYSTEM"].values[0]
        prep = Preporcessor(train_df)
        yaml_path = './test_data/model/configs/configs.yaml'
        model_list = ["AnomalyTransformer"]
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
        prep.PortSelection(port, lastdate=lastdate, lasttime=lastmin, padding =True) ### PADDING 말고 보간 기능으로 바뀜

        model_path, scaler_path ,z_path, df_path, Th_path, save_path_pkl,save_path_csv = PathCreator(args, 
                                                                                                     data_path, 
                                                                                                     model_type, 
                                                                                                     port_path, 
                                                                                                     col_path, 
                                                                                                     lastdate, 
                                                                                                     db, 
                                                                                                     table_path)

#             feature_list = "R_" + col if r_feature != "A" else col
        prep.LoadScaler(col=feature_list, scaler_type="Standard", path=scaler_path, save=False) ### False is important

        threshold = LoadThreshold(Th_path)

        _, test_df = TestADModel(model = model,## 모델 객체를 넘겨주기 for 루프 전에 생성
                        model_configs = model_configs, 
                        model_type = model_type, 
                        model_path = model_path,
                        n_splits=n_split,
                        data = prep ,col=col, 
                        device=device,
                        CV=True, 
                        epochs=10,
                        seq_len =seq_len,
                        feature_list=[feature_list],
                        partial_scaling=False)    

        x =test_df["SCORE"] #.values

        length = len(x)-args["seq_len"]+1
        prep.InverseScaler(feature_list, path=scaler_path) #inversed data is saved at {col}_ORIGIN column 
        test_df["DATETIME"] = test_df["DATETIME"][7:]
        test_df["SCORE"] = test_df["SCORE"][7:]
        test_df["DB"] = [db] *length
        test_df["TABLE"] = [table]*length
        test_df["COLUMNS"] = [col]*length
        test_df["SYSTEM"] = [system_name]*length
        test_df["PORT"] = [port]*length

        if r_feature != "A":
            test_df["VALUE"]=prep.port_df[col].reset_index(drop = True).values[7:] 
            test_df["R_VALUE"]=prep.port_df[f"{feature_list}_ORIGIN"].reset_index(drop = True).values[7:] 
        else:
            test_df["VALUE"] = prep.port_df[f"{col}_ORIGIN"].reset_index(drop = True).values[7:] 
            test_df["R_VALUE"]= ["None"]

        detection_list = [1 if score >= threshold else 0 for score in test_df['SCORE']]
        test_df['DETECTION'] = detection_list

        test_df["THRESHOLD"] = [threshold] * length
        Threshold_list = [abt if abt != np.inf else 10000 for abt in test_df['THRESHOLD']]
        test_df['THRESHOLD'] = Threshold_list

        test_df["ABNORMALITY"] = ((test_df["SCORE"]/threshold ) *100 )
        ABNORMALITY_list = [abt if abt != np.inf else 10000 for abt in test_df['ABNORMALITY']]
        test_df['ABNORMALITY'] = ABNORMALITY_list
        # print(test_df)

        if args['insert_sql']:
            InsertOuputData(test_df, db, lastdate, 'test_ai_results_v3')
            print("date insert is completed")
        del test_df, prep
        gc.collect()

            
        i += 1
        if i // 1000 == 0:
             gc.collect()



def process_inference(args,chunk):
    inference(args, chunk)

        
        
if __name__ == '__main__':

    mp.set_start_method('spawn')
    
    parser = argparse.ArgumentParser(description = "5G Core Predictive-Prevention")
    parser.add_argument("--server",type=str, default='server1') #server1 or server2
    parser.add_argument("--mp_processor",type=int, default=2)
    parser.add_argument("--model_path",type=str,default='/home/infra/Documents/')
    parser.add_argument("--checker_path",type=str,default ="./trainable_checker_v4.csv")
    parser.add_argument("--save_path",type=str,default="/home/infra/Documents/save_path")
    parser.add_argument("--GPU",type=str, default='cpu')
    parser.add_argument("--DB_set", type=str, default="")
    # parser.add_argument("--DB_set", type = list ,nargs='+', default="") 
    parser.add_argument("--seq_len",type=int, default=8)
    parser.add_argument("--insert_sql", action=argparse.BooleanOptionalAction, default=False)    
    parser.add_argument("--CV",type=int, default=1)
    args = parser.parse_args()
    args = vars(args)
    
    trainable_csv=pd.read_csv("./trainable_checker_v6.1.csv").drop_duplicates()
    trainable_csv = trainable_csv.loc[trainable_csv['AT_model'] ==True,:]
    trainable_csv_tmp1 = trainable_csv.set_index(["DB_name", "table_name", "port", "col_name","R_feature"]).sort_values(by=["DB_name", "table_name", "port"])
    
#     torch.set_flush_denormal(True)
    
    nf = {"SET_1": [ 'BS_AMF22_5M', 'BS_SMF02_5M', 'BS_SMF03_5M', 'BS_UPF01_5M', 'BS_UPF03_5M',
                     'DJ_UPF03_5M','DJ_AMF21_5M', 'DJ_AMF22_5M', 'DJ_SMF01_5M', 'GJ_UPF01_5M', 'DG_UPF01_5M'], ## 1차 학습
          "SET_2": ['BS_AMF21_5M', 'DJ_UPF01_5M'], # 2차 학습
          "SET_3": ['DJ_SMF03_5M'], #3차 학습
          "SET_4": ['GR_AMF21_5M', 'GR_AMF22_5M', 'GR_SMF01_5M', 'GR_SMF04_5M','GR_UPF01_5M','GR_UPF03_5M','GR_UPF05_5M',
                    'GR_UPF06_5M', 'HH_AMF21_5M','HH_AMF22_5M','HH_SMF01_5M','HH_SMF04_5M', 'HH_UPF01_5M','HH_UPF03_5M',
                    'HH_UPF04_5M','HH_UPF05_5M', 'HH_UPF06_5M'] # 4차 학습
         }
    train_list = []
    print("inference start")

    db_set = args["DB_set"][1:-1].strip("""'[""").strip("""]'""")
    db_set = db_set.split(""", """)
    print(args["DB_set"],db_set,db_set[0])
    print(type(db_set))

    for idx in trainable_csv_tmp1.index:
        if idx[0] in db_set: # ''DJ_SMF03_5M','
            train_list.append(idx)
          
    
    time1 = time.time()

    # print("hi","train_list",train_list)    
    db, table = train_list[0][0:2]
    args["last_time"] = GetLastTime(db,table)
    
    multi_procs = True
    n_procs = args["mp_processor"]
    # n_procs =1
    if n_procs ==1 :
        multi_procs = False

    if not multi_procs:
        inference(args, train_list)
        # cProfile.run('inference(args, train_list)')
        
    else: #140초
        num_chunk = len(train_list) // n_procs
        rest = len(train_list) % n_procs
        
        cell_chunks = []
        start = 0
        for i in range(n_procs):
            end = start + num_chunk + (1 if i < rest else 0)
            cell_chunks.append(train_list[start:end])
            start = end
            
#         chunks = distribute_groups(grouped, n_procs)
        
            processes = []
        

        for chunk in cell_chunks:
        # for chunk in chunks:
            p = mp.Process(target=process_inference, args=(args, chunk))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()

        print(f"Inference 실행 완료 시간 :{time.time()-time1}s")
