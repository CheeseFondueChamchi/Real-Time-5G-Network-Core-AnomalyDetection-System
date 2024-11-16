import pymysql
import yaml
import argparse
import getpass
from preprossesor import Preporcessor, dataload, train_folder_dir, CreatePath
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
# from sklearn.preprocessing import MinMaxScaler
import time
from threshold_func import find_threshold
from logger import timer
## v1 : Statical Learning Condition Changed - 2024-03-27  19:27:00
## v2 : Inference Mode with not consider mp - 2024-04-01
## v3 : 0514 insert DATA 


def PathCreator(args,csv_save_path,model_type,port_path,col_path,lasttime,db,table_name):
    basemodel_path = args["model_path"]
    save_path = args["save_path"]
    folder_name = lasttime.strftime("%Y_%m_%d_%H_%M_%S")
    model_path = os.path.join(f"{basemodel_path}/5Gcore_newyear3/model/{csv_save_path}",model_type,f'{port_path}_{col_path}.ckpt')
    scaler_path = os.path.join(f"{basemodel_path}/5Gcore_newyear3/test_result2/Scaler/{csv_save_path}",model_type,f'{port_path}_{col_path}.pkl')        
    z_path = os.path.join(f"{basemodel_path}/5Gcore_newyear3/test_result2/Stat/{csv_save_path}",model_type,f'{port_path}_{col_path}')                   
    df_path = os.path.join(f"{basemodel_path}/5Gcore_newyear3/test_result2/A_Score/{csv_save_path}",model_type,f'{port_path}_{col_path}.csv')
    Th_path = os.path.join(f"{basemodel_path}/5Gcore_newyear3/test_result2/Threshold/{csv_save_path}",model_type,f'{port_path}_{col_path}.pkl')
    save_path_pkl = os.path.join(f"{save_path}/test_result2/Threshold/{folder_name}/{csv_save_path}/",model_type,f'{port_path}.pkl')
    save_path_csv = os.path.join(f"{save_path}/test_result2/Threshold/{folder_name}/{csv_save_path}/",model_type,f'{port_path}.csv')
    return model_path, scaler_path ,z_path, df_path, Th_path, save_path_pkl,save_path_csv

def RulePathCreator(args,csv_save_path,model_type,port_path,col_path,lasttime,db,table_name):
    basemodel_path = args["model_path"]
    save_path = args["save_path"]
    folder_name = lasttime.strftime("%Y_%m_%d_%H_%M_%S")
    model_path = os.path.join(f"{basemodel_path}/5Gcore_newyear3/model/",model_type,f'{port_path}_{col_path}.ckpt')
    scaler_path = os.path.join(f"{basemodel_path}/5Gcore_newyear3/test_result2/Scaler/",model_type,f'{port_path}_{col_path}.pkl')        
    z_path = os.path.join(f"{basemodel_path}/5Gcore_newyear3/test_result2/Stat/",model_type,f'{port_path}_{col_path}')                   
    df_path = os.path.join(f"{basemodel_path}/5Gcore_newyear3/test_result2/A_Score/",model_type,f'{port_path}_{col_path}.csv')
    Th_path = os.path.join(f"{basemodel_path}/5Gcore_newyear3/test_result2/Threshold/",model_type,f'{port_path}_{col_path}.pkl')
    save_path = os.path.join(f"{save_path}/test_result2/Threshold/{folder_name}/{db}/{table_name}/",model_type,f'{port_path}.pkl')
    return model_path, scaler_path ,z_path, df_path, Th_path


def GetData(sql,db): #from my sql
    cnx = pymysql.connect(user='',password= '',host = '', database = db)
    cursor = cnx.cursor()
    cursor.execute(sql)
    tables = cursor.fetchall()
    cursor.close()
    cnx.close()
    return tables
def GetLastTime(db,table):
    
    try:
        cscf = "CSCF"
        if cscf in db:
            cnx = pymysql.connect(user='',password= '!',host = '', port = , database = db)
            cursor = cnx.cursor()
        else:
            cnx = pymysql.connect(user='',password= '!',host = '1', port = , database = db)
            cursor = cnx.cursor()

        sql =  f"""SELECT MAX(DATETIME) FROM `{table}`;"""
        cursor.execute(sql)
        tables = cursor.fetchall()[0][0]  

    except Exception as E:
        print("Extracting TCAS DB Lasttime while inference make error")
    finally:
        cursor.close()
        cnx.close()
    
    return tables

@timer
def GetTrainDataByDG( db, table,port,col,lastmin = 60*24*30): #from my sql
    end = GetLastTime(db,table)
    start = end - datetime.timedelta(minutes=lastmin)
    # print("train priod :",start,end )
    try:
        cscf = "CSCF"
        if cscf in db:
            cnx = pymysql.connect(user='',password= '!',host = '', port =  database = db)
            
        else:
            cnx = pymysql.connect(user='',password= '',host = '', database = db)
#
            
        sql = f"""SELECT DATETIME,PORT,SYSTEM, `{col}` FROM `{table}` where PORT = '{port}' and DATETIME BETWEEN '{start}' and '{end}'"""
        print(db,sql)
        # sql = f"""SELECT DATETIME,PORT,'SYSTEM', `{col}` FROM `{table}` WHERE DATETIME <= '{end}' and PORT = '{port}' ORDER BY DATETIME DESC LIMIT 8"""
        tables = pd.read_sql(sql, cnx)
    
    except Exception as E:
        print("Extracting TCAS DB inference DATA make error")
    finally:
        
        cnx.close()
        
    return tables, end
@timer
def GetTrainData( db, table,port,col,lastmin = 60*24*30): #from my sql
    end = GetLastTime(db,table)
    start = end - datetime.timedelta(minutes=lastmin)
    # print("train priod :",start,end )
    try:
        cscf = "CSCF"
        if cscf in db:
            cnx = pymysql.connect(user='',password= '!',host = '', port = , database = db)
            
        else:
            cnx = pymysql.connect(user='',password= '!',host = '', port = , database = db)
            
        sql = f"""SELECT DATETIME,PORT,SYSTEM, `{col}` FROM `{table}` where PORT = '{port}' and DATETIME BETWEEN '{start}' and '{end}'"""
        print(db,sql)
        # sql = f"""SELECT DATETIME,PORT,'SYSTEM', `{col}` FROM `{table}` WHERE DATETIME <= '{end}' and PORT = '{port}' ORDER BY DATETIME DESC LIMIT 8"""
        tables = pd.read_sql(sql, cnx)
    
    except Exception as E:
        print("Extracting TCAS DB inference DATA make error")
    finally:
        
        cnx.close()
        
    return tables, end


def find_prefix(table_name, col):
    ATTEMPT =   {  "SMDNN|FB": ["EPS", "IRAT", "NO_MM"],
    "RAD|ACCT": ["START", "STOP", "INTERIM"],
    #"POLEVT|DNN": ["NUM_OF_PACKET_FILTER", "RES_RELEASE", "SUCC_RES_ALLO"],
    "PFCP|PFMC": ["AS_S_REQ_RX", "AS_S_REQ_TX", "SE_REQ", "SM_REQ", "SD_REQ", "SR_REQ", "PFDM_REQ", "AS_U_REQ_RX", "AS_REL_REQ", "AS_U_REQ_TX", "ND_REQ"],
    "GTPC|TMMC": ["CRT_SESS", "CRT_BEAR", "MOD_BEAR", "UPD_BEAR", "DEL_SESS", "DEL_BEAR", "BEAR_RSC", "MOD_BR_CFI", "DEL_BR_CFI", "CRT_FWD", "DEL_FWD", "DEL_IND_FWD", "CRT_IND_FWD", "REL_ABEAR", "DNDATA_ACK", "DNDDATA_FI", "MOD_ABEAR","PDN_TRIG"],
    "DIAM|OCS": ["CCR_I", "CCR_U", "CCR_T", "RAR", "UNKN", "ASR"],
    "DIAM|PCRF": ["CCR_I", "CCR_U", "CCR_T", "RAR", "UNKN"],
    "COSVC|UDMUECM": ["REG", "DEREG", "DEREGNOTI", "UPDATE", "CSCFRSTO"],
    "COSVC|UDMSDM": ["GET", "SUBS", "UNSUBS", "NOTIFY", "INFO"],
    "COSVC|SMPOL": ["CREATE", "UPDATE", "DELETE", "NOTIFY"],
    "COSVC|AMFCOM": ["N1N2TRS", "N1N2FAIL", "N2NOTI", "EBIASSIGN"]
}
    REQ = {
        "DDN|PRI": ["DDN_REQ"],
        "DDN|PEER": ["DDN_REQ"],
        
    }
    POLEVT = {
        "POLEVT|DNN": ['ATTEMPT(count)', 'C_RATIO(%)']
    } 
    

    prefix = ""
    
    if table_name in ATTEMPT:
        prefixes = ATTEMPT[table_name]
        y = max([x for x in prefixes if x in col], key=len)
        prefix = y if y in col else ""
        
    elif table_name in REQ:
        print("hi")
        prefixes = REQ[table_name]
        y = prefixes[0]
        prefix = y if y in col else ""
        
    elif table_name in POLEVT:
        prefixes = POLEVT[table_name]
        y = prefixes[0]
        prefix = y if y in col else ""
    
    print(y,prefix)
    return prefix

@timer
def GetTrainData_R( db, table,port,col,r_feature,lastmin = 60*24*30): #from my sql

    
    end = GetLastTime(db, table)
    start = end - datetime.timedelta(minutes=lastmin)
    try:
        cscf = "CSCF"
        if cscf in db:
            cnx = pymysql.connect(user='', password='!', host='', port=, database=db)
        else:
            cnx = pymysql.connect(user='', password='!', host='', port=, database=db)

        prev=find_prefix(table, col)
        
        if r_feature == "B":
            sql = f"""SELECT DATETIME, PORT, SYSTEM, `{prev}_ATTEMPT(count)` ,`{col}` FROM `{table}` WHERE PORT = '{port}' AND DATETIME BETWEEN '{start}' AND '{end}'"""
        elif r_feature == "C":
            sql = f"""SELECT DATETIME, PORT, SYSTEM, `DDN_REQ(count)`,`{col}` FROM `{table}` WHERE PORT = '{port}' AND DATETIME BETWEEN '{start}' AND '{end}'"""
        elif r_feature == "D":
            sql = f"""SELECT DATETIME, PORT, SYSTEM, `ATTEMPT(count)`,`{col}` FROM `{table}` WHERE PORT = '{port}' AND DATETIME BETWEEN '{start}' AND '{end}'"""
        else:
            raise ValueError("Invalid r_feature value")

        tables = pd.read_sql(sql, cnx)
        
        tables["R_"+col] = tables.iloc[:,4]/tables.iloc[:,3]
        tables = tables.replace(np.inf, 100).replace(np.nan, 0)
        tables = tables.loc[:,["DATETIME","PORT","SYSTEM",col,"R_"+col]]
        
    except Exception as e:
        print("An error occurred while extracting TCAS DB inference data:", e)
    finally:

        cnx.close()


    return tables, end


def GetInferenceData_sql( db, lastdate ,table,col,port,retries=3,delay=0.1,lastmin = 40):
    end =lastdate
    start = end - datetime.timedelta(minutes=lastmin)
    # print("inference priod :",start,end )
    conn_time = time.time()
    cscf = "CSCF"
    if cscf in db:
        cnx = pymysql.connect(user='',password= '!',host = '', port = , database = db)

    else:
        cnx = pymysql.connect(user='',password= '!',host = '', port = , database = db)

    sql = f"""SELECT DATETIME,PORT,SYSTEM, `{col}` FROM `{table}` where PORT = '{port}' and DATETIME BETWEEN '{start}' and '{end}'  """
    try:
        with cnx.cursor() as cursor:
            for attempt in range(retries):
                try:
                    cursor.execute(sql)
                    result = cursor.fetchall()
                    columns = [col_desc[0] for col_desc in cursor.description]
                    return pd.DataFrame(result, columns = columns), lastdate 
                except (pymysql.err.OperationalError, pymysql.err.InternalError) as e:
                    print(f"Error: {e}, retrying in {delay} seconds at {end}")
                    time.sleep(delay)
                finally:
                    cnx.commit()
    finally:
        cnx.close()
          
@timer
def GetInferenceTableData( db,port ,lastdate ,table,lastmin = 40): #from my sql
    end =lastdate
    start = end - datetime.timedelta(minutes=lastmin)
    # print("inference priod :",start,end )
    try:
        cscf = "CSCF"
        if cscf in db:
            cnx = pymysql.connect(user='',password= '!',host = '', port = , database = db)
            
        else:
            cnx = pymysql.connect(user='',password= '!',host = '', port = , database = db)
            

        sql = f"""SELECT * FROM `{table}` where DATETIME BETWEEN '{start}' and '{end}' """
        tables = pd.read_sql(sql, cnx)

 
    except Exception as E:
        print("Extracting TCAS DB inference DATA make error")
    finally:
        
        cnx.close()
        
    return tables, lastdate

@timer
def GetInferenceData( db, lastdate ,table,col,port,lastmin = 40): #from my sql

    end =lastdate
    start = end - datetime.timedelta(minutes=lastmin)
    # print("inference priod :",start,end )
    try:
        cscf = "CSCF"
        if cscf in db:
            cnx = pymysql.connect(user='',password= '!',host = '', port = , database = db)
            
        else:
            cnx = pymysql.connect(user='',password= '!',host = '', port = , database = db)
            

        sql = f"""SELECT DATETIME,PORT,SYSTEM, `{col}` FROM `{table}` where PORT = '{port}' and DATETIME BETWEEN '{start}' and '{end}'    """
        tables = pd.read_sql(sql, cnx)

 
    except Exception as E:
        print("Extracting TCAS DB inference DATA make error")
    finally:
        
        cnx.close()
        
        
    return tables, lastdate

@timer
def GetInferenceData_R( db, lastdate ,table,col,port,r_feature,lastmin = 40): #from my sql

    end =lastdate
    start = end - datetime.timedelta(minutes=lastmin)

    try:
        cscf = "CSCF"
        if cscf in db:
            cnx = pymysql.connect(user='', password='!', host='', port=, database=db)
        else:
            cnx = pymysql.connect(user='', password='!', host='', port=, database=db)

        prev=find_prefix(table, col)
        
        if r_feature == "B":
            sql = f"""SELECT DATETIME, PORT, SYSTEM, `{prev}_ATTEMPT(count)` ,`{col}` FROM `{table}` WHERE PORT = '{port}' AND DATETIME BETWEEN '{start}' AND '{end}'"""
        elif r_feature == "C":
            sql = f"""SELECT DATETIME, PORT, SYSTEM, `DDN_REQ(count)`,`{col}` FROM `{table}` WHERE PORT = '{port}' AND DATETIME BETWEEN '{start}' AND '{end}'"""
        elif r_feature == "D":
            sql = f"""SELECT DATETIME, PORT, SYSTEM, `ATTEMPT(count)`,`{col}` FROM `{table}` WHERE PORT = '{port}' AND DATETIME BETWEEN '{start}' AND '{end}'"""
        else:
            raise ValueError("Invalid r_feature value")

        tables = pd.read_sql(sql, cnx)
        
        tables["R_"+col] = tables.iloc[:,4]/tables.iloc[:,3]
        tables = tables.replace(np.inf, 100).replace(np.nan, 0)
        tables = tables.loc[:,["DATETIME","PORT","SYSTEM",col,"R_"+col]]
        
    except Exception as e:
        print("An error occurred while extracting TCAS DB inference data:", e)
    finally:
        cnx.close()
        
    return tables, lastdate

def rename_keys(dictionary, key_mapping):
    renamed_dict = {}
    for key, value in dictionary.items():
        new_key = key_mapping.get(key, key)
        renamed_dict[new_key] = value
    return renamed_dict

@timer
def InsertOuputData(df, db, lastdate, table_name):
    # try:
    df = rename_keys(df, {'DATETIME': 'original_datetime',
                       'DB': 'db_name',
                       'TABLE': 'table_name',
                       'PORT': 'port',
                       'SCORE': 'score',
                       'COLUMNS': 'statistic',
                       'DETECTION': 'detection',
                       'ABNORMALITY': 'abnormality',
                       'THRESHOLD': 'threshold',
                       'VALUE': 'statistic_value',
                        'R_VALUE': 'statistic_ratio_value',
                       'SYSTEM': 'nf_name'})

    # df = df.loc[df["original_datetime"]==lastdate,:]

    # Connect to the database
    cnx = pymysql.connect(host = '',port=,user='',password= '!', database = '',charset = '')  # Replace with actual code to connect to your database

    ##----------------------------------------------------------------
    # Create a cursor object
    cur = cnx.cursor()
    # print("before loop",*df.values())
    
    for  row in zip(*df.values()):
        # print("after loop",row)
        if row[8] == "None":
            statistic_ratio_value = None
        else:
            statistic_ratio_value = float(row[8])
        sql = f"""INSERT INTO {table_name}(`original_datetime`, `db_name`, `table_name`, `port`, `statistic`, `nf_name`, `statistic_value`, `statistic_ratio_value`,`score`, `detection`, `threshold`, `abnormality`) values (%s,%s,%s, %s,%s,%s,%s, %s,%s,%s, %s,%s) """
        # print(statistic_ratio_value)
        cur.execute(sql, (str(row[0]), str(row[2]), str(row[3]), 
                          str(row[6]),  str(row[4]),str(row[5]), 
                          float(row[7]),statistic_ratio_value, float(row[1]), int(row[9]), 
                         float(row[10]) , float(row[11]) ) )
        
    # Commit the changes to the database
    cnx.commit()
    cur.close()
    cnx.close()
    print("data is inserted")

    return

# @timer
def Thresholding(x): #x : variance value

    v  = np.var(x)
    q1, q3 = np.percentile(x,[25,75])
    iqr = q3 - q1
    upper_bound = q3 + (iqr * 1.5)
    print("np.var(x), upper bound",v,upper_bound)

    x_out_idx = np.where((x>upper_bound))[0]
    # print("Outlier 수",len(x[x_out_idx]))

    x_idx = np.where((x<upper_bound))[0]
    x = x[x_idx]
    v = np.var(x)
    a = plt.hist(x, bins=100,cumulative=True, edgecolor = 'white',linewidth= 5.8, label='Cumulative',density=True)
    if  v < 3.0:
        return v, 0.995, a
    elif 3.0 <= v :
        return v, 0.999 , a
def SaveThreshold(date,x,col,pb,Th_path,cho = True):
    if cho == True:
        threshold_2, best_rmse = find_threshold(date, x, col, z_score_threshold=3.715, fig_flag = 0)
        with open(Th_path, "wb") as f:
            pickle.dump(threshold_2+0.0000001, f)
        return threshold_2
    else:  
        x = np.percentile(x, 100 * (1 - pb))
        x = round(x,8)
        with open(Th_path, "wb") as f:
            pickle.dump(x, f)
        return x
    
def  LoadThreshold(Th_path,load_th=True):
    load_th = True
    if load_th == True:
        # print(Th_path)
        with open(Th_path, 'rb') as f:
            threshold = pickle.load(f)
    else:
        threshold = 1
    return threshold
    




# @timer
def InferenceDirCreator(args, data_path,model_list,lasttime):
    model_path = args["model_path"]
    save_path = args["save_path"]
    # print(model_path,save_path)
    
    for i in model_list:
        folder_name = lasttime.strftime("%Y_%m_%d_%H_%M_%S")
        path = os.path.join(f"{save_path}/test_result2/Threshold/{folder_name}/",data_path,i)
        CreatePath(path)


@timer
def TrainDirCreator(args, data_path,model_list,lasttime):
    model_path = args["model_path"]
    save_path = args["save_path"]
    print(model_path,save_path)
    
    for i in model_list:
        path = os.path.join(f"{model_path}/5Gcore_newyear3/test_result2/Threshold/",data_path,i)
## Threshold Save Path
        CreatePath(path)
        path = os.path.join(f"{model_path}/5Gcore_newyear3/test_result2/Scaler/",data_path,i) 
        CreatePath(path)
        path = os.path.join(f"{model_path}/5Gcore_newyear3/test_result2/Stat/",data_path,i) ## Z_Score Save Path
        CreatePath(path)
        path = os.path.join(f"{model_path}/5Gcore_newyear3/test_result2/A_Score/",data_path,i) ## Anomaly Score Save Path
        CreatePath(path)
        path = os.path.join(f"{model_path}/5Gcore_newyear3/model/",data_path,i) ## Trained Model Save Path
        CreatePath(path)


##예외 모델:
##Condition 에 따라 학습 X -> 이후 Rule 처리

#### 미 추론 데이터 check  str('datetime') transformes to datetime(ns) YYYY-mm-dd HH:mm:ss          
def SavingTimeLocal(args,trainable_df, lasttime,table,port,db): # saves the inference time in local as ----.csv
#     print("checktable,port,db",table,port,db)
    
    if "DATETIME" in trainable_df.columns:
        trainable_df['DATETIME'] = pd.to_datetime(trainable_df['DATETIME'])
#         print(type(lasttime))
        if trainable_df['DATETIME'].max() < lasttime:  # 첫 번째 요소와 datetime 비교
            trainable_df['DATETIME'] = lasttime  # datetime이 더 크면 datetime으로 교체
            trainable_df["Infered"] = False  # 날짜가 업데이트되면 Infered 컬럼은 다시 False로 설정
        else:
            print("datetime not yet updated")                                                 
    else : #trainable_df에 datetime column 이 없다면 추가 

        trainable_df["DATETIME"]=lasttime

    indices = trainable_df.loc[((trainable_df['DB_name']==db)&(trainable_df['port']==port)&(trainable_df['table_name']==table))].index
#     print("indicede",indices)
    trainable_df.loc[indices, "Infered"] = True ### trainable_df 내 columns PORT와 TABLE,db값이 일치하는 곳만 True
    print("not yet infered :",np.sum(trainable_df["Infered"] == False))
    trainable_df.to_csv(args["checker_path"],index=False)
    return trainable_df
    
def inference( col,prep,scaler_type,path,model_configs,model_type,model_path,n_splits,device,seq_len=8):
    feature_list= col                        
    prep.Scaler( col=col, scaler_type = "Standard", path= scaler_path,save=False)                        
    try:
        dicting, test_df = TestModel(model_configs = model_configs, 
            model_type = model_type, 
            model_path = model_path,
            n_splits=n_splits,
            data = prep ,col=col, 
            device=device,
            CV=True, 
            epochs=10,
            seq_len =seq_len,
            feature_list=feature_list)

        return dicting, test_df.iloc[:,:], False
    
    except FileNotFoundError:
        print("File not Found. Please provide the correct file address")
        return _, _,True 
#         continue
    except torch.serialization.UnpicklingError:
        print("Error loading the model file. Please make sure model states")
        return _, _,True 
#         continue
    except PermissionError:
        print("Permission denied.")
        return _, _,True 
#         continue
    except OSError as e:
        if e.errno == 13:
            print("File is being used by another process")
            return _, _,True 
#             continue
    except RuntimeError as e:
        print(f"Error reading file : {e}")
#         continue    
        return _, _,True 
