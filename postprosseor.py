import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
import os
from preprossesor import temporalize, reverse_temporalize, input_shaping, CreatePath
from datetime import datetime
import matplotlib.dates as mdates

#Read specific col of port_df and read index has anomaly values from TestMoModel result 
# TestModel Result has AS or Bools with CV columns 
# This will show the window and save the window 
# we need read the path for the result with port and column name 



# df: Test Model Result 
# path : save path 
# Window Interval = -20 + seq_len : 2*seq_len+20 
# 

def WindowCapture(df, path, seq_len, date_list,col ,threshold): 

    fig = plt.figure(figsize=(30, 25))
    gs = gridspec.GridSpec(nrows=5, ncols=1,height_ratios=[1,1,1,1,1])
    # print(date_list)
    
    #print(abnormal_time)
    df.sort_values(by='DATETIME',inplace = True)
    # print(df.loc[:,[col,f"{col}_ORIGIN","SCORE",f"{col}_Z_Score"]])
    x = df.loc[:,["VALUE","SCORE"]].values
    # x = df.loc[:,[col,"SCORE",f"{col}_Z_Score"]].values
    y = df.loc[:,"DATETIME"].values
    #print(x)

    lenght =  int(len(y)/2)-20 #int(len(y)) #
    ax0 = plt.subplot(gs[0])
    ax0.plot(y[:],x[:,0],color = "black", label = col)
    for date in date_list:
        plt.axvspan(date-np.timedelta64(seq_len*5,'m'),date,color='#FFA07A',alpha=0.3)

    ax3 = plt.subplot(gs[3])
    ax3.plot(y[:],x[:,1],color = "red", label = "SCORE")
    for date in date_list:
        plt.axvspan(date-np.timedelta64(seq_len*5,'m'),date,color='#FFA07A',alpha=0.3)
    
    plt.axhline(threshold, color='violet', linestyle='--', linewidth = 4 )
    

    ax0.set_title(f"{col}",fontsize=10, color = 'black')
    ax3.set_title("REAL SCORE",fontsize=10, color = 'black')

    ax0.set_facecolor('lightgrey')
    ax3.set_facecolor('lightgrey')
    # ax4.set_facecolor('lightgrey')
    path = path[:-4]+".png" 
    print(f"{path}를 확인해보자구나")  
    plt.savefig(path,format='png', dpi=150)  #파일명이 너무 길면 저장 못함


def AnomalyGraphCapture( df, col, result_col,path, seq_len, result_type= "AS",threshold = 0.5):
    df = df.reset_index(drop=True)
    if result_type == "AS":
        abnormal_date_list = df.loc[df[result_col]>=threshold, "DATETIME"].values
        
    #    print(abnormal_date_list)
        WindowCapture(df, path, seq_len, abnormal_date_list,col,threshold)

    elif result_type == "Bools":
        ## later threshold will be changed dinamicly to follow Probability Density Function
        abnormal_date_list = df.loc[df[result_col]==1, "DATETIME"].values
        if len(abnormal_date_list) > 0:
            WindowCapture(df, path, seq_len, abnormal_date_list,col)
        else:
            print("No Problem")


def AbnormalityReport(df, path):
    print("")


if __name__=='__main__':
    
    #tmp path
    port_name = ""
    tmp_model_path = port_name.replace("/","_")
    tmp_model_type = "AnomalyTransformer"

    path = f"./test_data/model/test/{model_type}/{model_path}_{col}.csv"



