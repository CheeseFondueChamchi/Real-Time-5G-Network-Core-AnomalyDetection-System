### data load module ## 아직까지 sql 생각은 고려 안해도 괜찮을 듯\
### 
import pandas as pd
import numpy as np
import os
# from datetime import datetime
import copy
import glob
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, RobustScaler
import datetime
from collections import OrderedDict
import torch
import random
from logger import timer

@timer
def seed_everything(seed):
    torch.manual_seed(seed) #torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다 
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) #numpy를 사용할 경우 고정
    random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정
    torch.manual_seed(seed)
    
def dataload(nf_name,dir): ## NF name, 개별 파일 # Port관리
    
    df_temp = pd.read_csv(dir, nrows=0)
    column_names = df_temp.columns
    # pprint.pprint( f"""{dir}'s column_name : """, column_names)
    print(f"{dir}'s column_name :  {column_names}")

    # datetime, port 제외한 데어터 열을 특정 타입(예: float16)으로 설정 
    dtype_dict = {col: 'float32' for col in column_names[4:] }
    df = pd.read_csv(dir, dtype=dtype_dict)
    # 설정된 dtype으로 전체 파일 읽기
    if 'Unnamed: 0' in df.columns: 
        df = pd.read_csv(dir, dtype=dtype_dict, index_col=0)
        print("This CSV File Has a Unnamed Column.")
    print(df.head())
    print(df.info())
    return df
def is_valid_datetime(dt):
    try:
        datetime.datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
        return True
    except ValueError:
        return False

def train_folder_dir(dir):
    directory_path = dir
    try:
        csv_files = glob.glob(os.path.join(directory_path,"**", "*.csv"),recursive=True)
        print(os.path.join(directory_path,"**.csv"))
        # csv_files = glob.glob(os.path.join(directory_path,"**.csv"),recursive=True)
        
    except Exception as e:
        print(f'{e} is happend. Can Not Find Out the CSV File.')
    return csv_files

# @timer
def merge_dict_lists(dict1, dict2):
    merged_dict = OrderedDict(dict1)

    for key, value in dict2.items():

        if key in merged_dict:
            merged_dict[key] = np.hstack([merged_dict[key],value])
        else:
            merged_dict[key] = value
    return merged_dict


def CreatePath(file_path):
#     print('Check Path....')
    try:
        if not os.path.exists(file_path):
            os.makedirs(file_path)
#             print(f"{file_path} is created. ")
    except Exception as e:
        print(f"{e}")

### 전처리 학습용 / Test 용
### Conduct null space interpolation / noise add or remove or keep
class Preporcessor: 
    def __init__(self, df): # data load and data null check and interpolation by column
       
        null_percent = df.isna().mean().values * 100
        # print(f"Data Null States:", null_percent)
        ### 포트별로 
        self.port_list = df['PORT'].unique()
        self.df = df

    def PortSelection(self, port, lastdate, lasttime, startdate=None, padding=False):
        # 데이터 정렬 및 필터링
        self.port_df = self.df.loc[self.df['PORT'] == port, :].reset_index(drop=True).sort_values(by='DATETIME')

        if padding:
            lasttime = lasttime  # 35 for inference 8 timestamp

            if isinstance(lastdate, datetime.datetime):
                if startdate is None:
                    startdate = lastdate - datetime.timedelta(minutes=lasttime - 5)

                # 생성할 datetime 인덱스 범위 설정
                datetime_index = pd.date_range(end=lastdate, start=startdate, freq='5T')

                # datetime 형식으로 변환
                self.port_df["DATETIME"] = pd.to_datetime(self.port_df["DATETIME"])
                self.port_df = self.port_df.set_index('DATETIME', drop=True)

                # float 타입 컬럼만 선택
                column_list = [column for column in self.port_df.columns if self.port_df[column].dtype == 'float']

                # 재인덱싱 (결측치가 있는 datetime 인덱스 생성)
                reindexed_df = self.port_df.reindex(datetime_index, fill_value=np.nan).sort_index(ascending=True)

                # PORT와 SYSTEM 값을 먼저 패딩으로 채움
                reindexed_df["PORT"] = reindexed_df["PORT"].fillna(method="bfill").fillna(method="ffill")
                reindexed_df["SYSTEM"] = reindexed_df["SYSTEM"].fillna(method="bfill").fillna(method="ffill")

                # 보간이 필요한 경우에만 보간 적용
                if reindexed_df[column_list].isna().sum().sum() > 0:
                    # 결측치가 있는 부분에만 선형 보간을 적용
                    reindexed_df[column_list] = reindexed_df[column_list].interpolate(method='linear', limit_area='inside')

                # 보간 후 남은 결측치는 패딩으로 처리
                reindexed_df[column_list] = reindexed_df[column_list].fillna(method="bfill").fillna(method="ffill")

                # 최종 결과
                self.port_df = reindexed_df.reset_index(drop=False)
                self.port_df = self.port_df.rename(columns={"index": "DATETIME"})
    def Interpolation(self, lastdate,lasttime,interval = "min" ): ### PortSelectioN이 선언 되어야 이후
        # print("Padded df Shape :",self.port_df.shape)
        print("Interpolation 수행")
        lasttime = lasttime # 35 for inference 8 timestamp
        if isinstance(lastdate, datetime.datetime) :

            print("Interpolation DATETIME Check")
            start = lastdate - datetime.timedelta(minutes=lasttime-5)
            datetime_index = pd.date_range(end=lastdate, start= start, freq = interval)
            print(datetime_index)
            print("port_df",self.port_df)
            print(self.port_df.info())

            self.port_df["DATETIME"] = pd.to_datetime(self.port_df["DATETIME"])
            self.port_df = self.port_df.set_index('DATETIME', drop=True)
            column_list = [column for column in self.port_df.columns if self.port_df[column].dtype == 'float32']
            print(column_list)

            #column_list = col

            reindexed_df = self.port_df.reindex(datetime_index, fill_value=np.nan).sort_index(ascending=True)                     # Fill missing values with zeros       
            reindexed_df["PORT"] = reindexed_df["PORT"].fillna(method="bfill", axis=0)  # 이전 값으로 채우기
            reindexed_df["PORT"] = reindexed_df["PORT"].fillna(method="ffill", axis=0)
            reindexed_df["SYSTEM"] = reindexed_df["SYSTEM"].fillna(method="bfill", axis=0)  # 이전 값으로 채우기
            reindexed_df["SYSTEM"] = reindexed_df["SYSTEM"].fillna(method="ffill", axis=0)            
            reindexed_df[column_list]= reindexed_df[column_list].astype(np.float64)                
            reindexed_df[column_list] = reindexed_df[column_list].interpolate(method='polynomial',order =3)
            reindexed_df[column_list]= reindexed_df[column_list].astype(np.float32)
            print(reindexed_df.head())
            reindexed_df[column_list] = reindexed_df[column_list].fillna(method="bfill", axis=0)  # 이전 값으로 채우기
            reindexed_df[column_list] = reindexed_df[column_list].fillna(method="ffill", axis=0)

            nan_values = reindexed_df.columns[reindexed_df.isna().any()]
            print("Nan 값 : ",nan_values)
            print(nan_values)

            self.port_df = reindexed_df.reset_index(drop=False) # if col not in nan_values else self.port_df
            self.port_df = self.port_df.rename(columns={"index":"DATETIME"})
            print("Interpoled_DF Shape :",self.port_df)

        # except Exception as e: 
        #     print('예외 발생. 기존 port_df 사용', e)

    @timer
    def Scaler(self, col, scaler_type = "Robust", path= "./",save=False):
        try:
            print(scaler_type)
            if scaler_type == "Robust": ## 이상치에 덜 민감
                # print(f"{scaler_type}이 선택되었습니다. ")
                scaler = RobustScaler()
            elif scaler_type == "Standard": ## 평균적
                # print(f"{scaler_type}이 선택되었습니다. ")
                scaler = StandardScaler()
            elif scaler_type == "MinMax": ## 이상치에 민감 
                # print(f"{scaler_type}이 선택되었습니다. ")
                scaler = MinMaxScaler()
            elif scaler_type == "Normalizer":
                # print(f"{scaler_type}이 선택되었습니다. ")
                scaler = Normalizer()
            self.port_df.loc[:,col] \
                = scaler.fit_transform(self.port_df.loc[:,col].values.reshape(-1,1))
        except Exception as e:
            print(e)
            print("Scaler 동작 안함")
        if save:            
            with open(path, "wb") as f:
                pickle.dump(scaler, f)
                # print(f"해당 경로에 Scaler를 저장하였습니다 : {path}")  
                
    def Diff(self, col, n_diff):
        for i in range(n_diff):
            self.port_df[f'col_{i+1}_diff'] = self.port_df[col].diff(i+1).fillna(method="bfill")





    # @timer
    def LoadScaler(self, col, scaler_type="Robust", path="./", save=False):

        if os.path.exists(path):
            with open(path, "rb") as f:
                scaler = pickle.load(f)
                # print(f"Scaler를 {path}에서 불러왔습니다.")
                self.port_df.loc[:,col] = scaler.transform(self.port_df.loc[:,col].values.reshape(-1,1))
        else:
            pass

    # @timer
    def InverseScaler(self, col, path= "./"):
        with open(path, "rb") as f:
            scaler = pickle.load( f)
            # print(f"해당 경로에 Scaler를 불러왔습니다 : {path}")  
            
        self.port_df[f"{col}_ORIGIN"] = scaler.inverse_transform(self.port_df.loc[:,col].values.reshape(-1,1))

        
    def NoiseKeep(self):
        print(f"Noise_변환 없음을 적용합니다.")    
        self.port_df = self.port_df
    
    def NoiseGaussian(self, npower):
        print(f"Noise_gaussian을 적용합니다.") 
        for col in self.port_df.columns:
            if self.port_df[col].dtype == 'float32': # float16 can makes nan with np's methods
                std = np.asarray(self.port_df[col], dtype=np.float32).std()
                noise = np.random.normal(0, std * npower, size=len(self.port_df[col]))
                print(self.port_df[col][70:80], noise, std)
                self.port_df[col] += noise
    # moving ['sum' 'mean' 'min' 'max']처리, 
    # Input : num=5, method = "sum",feature_list = ['col1','col2','col3'] 
    def Rolling(self, length = 3,method = 'sum'):
        print(f"Noise_remove is applied with Pandas Rolling (Length : {length}, method : {method}")    
        print("This method must be used before taking 'DayToNum' method ")
        for col in self.port_df.columns:
            if self.port_df[col].dtype != 'object':
                rolled_series = getattr(self.port_df[col].rolling(window=length), method)()
                self.port_df[col] = rolled_series.fillna(method='ffill')
    def DayToNum(self):
        # Datatime index would have mixed formats like "2022-10-19 00:00:00" or "2022-10-19" then, we should adapt the datetime format as 'ISO8601'
        self.port_df['DATETIME'] = pd.to_datetime(self.port_df['DATETIME'])
        def convert_datetime(row):
            # 요일 변환 (월요일: 1, 일요일: 7)
            day_of_week = row.weekday() + 1
            # 시간 변환 (00:00 - 00:04 : 1, 23:55 - 23:59 : 288) 
            # Scale changed 1~288 to 1/288 ~ 288~288 
            time_of_day = ((row.hour * 60 + row.minute) // 5 + 1)/100
            return day_of_week, time_of_day      
        ## 20xx-xx-xx xx:xx:xx XXXX(요일) to 1~288(시간), 1~7(요일)   
        self.port_df['DAY_OF_WEEK'], self.port_df['TIME_OF_DAY'] = zip(*self.port_df['DATETIME'].apply(convert_datetime))
        self.port_df = self.port_df.sort_values(by='DATETIME').reset_index(drop=True)
        # print("DayToNum",self.port_df.head())

    def Sampling(self, over=True):
        df_next = self.port_df.shift(-1)
        # 'DATETIME' 열을 제외하고 계산
        numeric_columns = self.port_df.select_dtypes(include=[np.number]).columns
        middle_df = (self.port_df[numeric_columns] + df_next[numeric_columns]) / 2
        # 'DATETIME', 'DAY_OF_WEEK', 'TIME_OF_DAY', 'PORT' 열은 원본 값을 유지
        for col in ['DATETIME', 'DAY_OF_WEEK', 'TIME_OF_DAY', 'PORT']:
            if col in self.port_df.columns:
                middle_df[col] = self.port_df[col]
        # 마지막 행은 중간값 계산에서 제외
        middle_df = middle_df.iloc[:-1]
        # 원본 데이터프레임과 중간값 데이터프레임 병합
        self.port_df = pd.concat([self.port_df, middle_df]).sort_index(kind='merge').reset_index(drop=True)
        # print(self.port_df.head())
#####

def temporalize(data,lookback, feature): ## port_df를 numpy input
    print("Temporalization with DataFrame to Numpy")
    input_x = data.loc[:,feature].values
    X = np.reshape(input_x, (-1,len(feature)))
    output_X = []
    # output_y = []
    for i in range(np.shape(X)[0]-lookback+1):
        t = []
        # p = []
        t.append(X[i:i+lookback,:])
        # p.append(y[i:i+lookback,:])
        output_X.append(t)
        # output_y.append(p)
    return input_shaping(np.array(output_X),lookback,len(feature)) #, np.array(output_y)

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

    
if __name__ == '__main__':
    nf_name = "SMF"
    df = dataload(nf_name,"./test_data/HHSMF01_CFR_CFRI.csv") ## NF name, 개별 파일 # Port관리
    
    path = './test_data/model/configs/configs.yaml'
    
    # 전치리 모듈
    prep = Preporcessor(df)
    for tmp_port in prep.port_list[3:4]:
        prep.PortSelection(tmp_port, padding = True) #port, datetime = self.today,  
        print(f"NF : {nf_name}, Port Name : {tmp_port}, Port_df : {np.shape(prep.port_df)}")
        print("Pre-Processing Is Start")
        prep.Rolling(num=5, method = "sum")
        prep.NoiseGaussian(0.01)
        prep.Sampling()
        prep.DayToNum() # port_df 대상만 함
        print("Pre-Processing Is End")
        print(f"NF : {nf_name}, Port Name : {tmp_port}, Port_df : {np.shape(prep.port_df)}")
