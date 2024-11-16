from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import numpy as np; import pandas as pd
from pandas import Series
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.dates as mdates
import datetime; import time; import sys; import os, glob; import re

def moving_average(data):
    n = len(data)
    moving_avg = []
    
    for i in range(n):
        if i == 0:
            avg = data[i]
        else:
            avg = (data[i] + data[i-1]) / 2
        moving_avg.append(avg)
    
    return moving_avg

def calc_slope(y):
    clean_y = y[~np.isnan(y)]
    if len(clean_y) < 2:
        return np.nan
    X = np.arange(len(clean_y)).reshape(-1,1)
    model = LinearRegression().fit(X, clean_y)
    return model.coef_[0]

def threshold_transform(data, threshold, scale):
    transformed_data = []
    for value in data:
        if value > threshold*scale:
            # transformed_value = threshold*scale + ((value - threshold*scale) ** 1/400)
            transformed_value = threshold*scale + ((value - threshold*scale) ** 1/300)
            transformed_data.append(transformed_value)
        else:
            transformed_data.append(value)
    return transformed_data

def make_std_based_weight(data):
    std_based_weight = np.zeros_like(data)

    for i in range(len(data)):
        start_index = max(0, i-8)
        end_index = max(0, i)
        window = data[start_index:end_index-1]
        # print(len(window))
        mean = np.mean(window)
        std_dev = np.std(window)
        # print(mean, std_dev)
    
        # if std_dev != 0:
        if std_dev >= 0.0001:
            std_based_weight[i] = (data[i] - mean) / std_dev
        elif std_dev < 0.0001 and abs(data[i] - mean) > 0.0001:
            std_based_weight[i] = 8 * abs(data[i] - mean)
        else:
            std_based_weight[i] = 1
            
    return std_based_weight

def plot_data(df, y_data_0, y_data_1, start_plot, end_plot, threshold, y_right_min, y_right_max, nf_name):
    x_data = df['DATETIME']
    # y_data_0 = df[col_name_0]
    # y_data_1 = df[col_name_1]

    title = nf_name + ' | ' + df['PORT'][0] + ' | ' + df['COLUMNS'][0]

    fig, ax1 = plt.subplots(figsize=(15, 8))
    # fig, ax1 = plt.subplots(figsize=(10, 6))

    # Formatting the x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    ax1.tick_params(axis='x', labelsize=14)  # Adjust the font size of the datetime labels on the x-axis

    # Plot traffic data on the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Time', color='black', fontsize=16)
    ax1.set_ylabel(df['COLUMNS'][0], color='black', fontsize=16)
    ax1.plot(x_data, y_data_0, color=color, marker='o', markersize=5)
    line1, = ax1.plot(x_data, y_data_0, color=color, label=df['COLUMNS'][0])  # Add label for legend
    ax1.tick_params(axis='y', labelcolor='black', labelsize=16)
    if "RATIO(%)" in df['COLUMNS'][0]:
        ax1.set_ylim(95, 101)
    else:
        ax1.set_ylim(0, y_data_0.max()*1.3)  # Adjust as needed based on your data
    ax1.set_xlim(pd.to_datetime(start_plot), pd.to_datetime(end_plot))

    # Create a second y-axis for the anomaly scores
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Anomaly Score', color='black', fontsize=16)
    ax2.plot(x_data, y_data_1, color=color, marker='o', markersize=5)
    line2, = ax2.plot(x_data, y_data_1, color=color, label='Anomaly Score')  # Add label for legend
    ax2.tick_params(axis='y', labelcolor='black', labelsize=15)
    ax2.set_ylim(y_right_min, y_right_max)
    ax2.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label='Threshold')

    # Collecting handles and labels for the legend from both axes
    handles, labels = [], []
    for ax in [ax1, ax2]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles.append(handle)
            labels.append(label)

    # Add a single legend with both entries
    ax1.legend(handles, labels, loc='upper right', fontsize=16)

    # Add the title to the plot
    ax1.set_title(title, fontsize=18)

    # Improve layout and show plot
    fig.tight_layout()
    # plt.grid(True, axis='both', color='grey', alpha=0.2, linestyle='--')
    plt.grid(which='both', axis='both', color='grey', alpha=0.2, linestyle='--')
    plt.show()
    
    # Save the plot as an image
    # plt.savefig(title+'.png')
    
def find_threshold(training_data, anomaly_scores, col_name, z_score_threshold = 3.715, fig_flag = 1):

    ### Pre-processing fo Robust Estimation of PDF
    anomaly_scores_no_extream_value = anomaly_scores[anomaly_scores < 1] ## Removal of upper ultra-extreme values frequently occurring in RATIO statistics
    anomaly_scores_no_extream_value = anomaly_scores_no_extream_value[anomaly_scores_no_extream_value > 0.00000001] ## Removal of loert ultra-extreme values
    if len(anomaly_scores_no_extream_value) < 100:
        print("데이터가 이상")
        anomaly_scores_no_extream_value = anomaly_scores
    cut_off_pos = np.percentile(anomaly_scores_no_extream_value, 99.0)  ## Conventional outlier removal
    filtered_anomaly_scores = anomaly_scores_no_extream_value[anomaly_scores_no_extream_value <= cut_off_pos]  ## Conventional outlier removal
    filtered_anomaly_scores = filtered_anomaly_scores[filtered_anomaly_scores != 0]   ## Special outlier removal

    ### Setting the nunber of histogram bins and defining PDF family set
    # bins = 1000  # RATIO
    bins = int(len(filtered_anomaly_scores)/6)
    hist, bin_edges = np.histogram(filtered_anomaly_scores, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    dist_names = ['norm', 'lognorm', 'exponnorm', 'powernorm', 'exponweib', 'gamma', 'beta', 'expon']
    dist_functions = [stats.norm, stats.lognorm, stats.exponnorm, stats.powernorm, stats.exponweib, stats.gamma, stats.beta, stats.expon]

    ### Optimal PDF Estimation
    best_dist = None; best_params = None; best_rmse = np.inf

    for dist_name, dist_func in zip(dist_names, dist_functions):
        try:
            params = dist_func.fit(filtered_anomaly_scores)
            pdf = dist_func.pdf(bin_centers, *params)
            hist = hist/sum(hist)
            pdf = pdf/sum(pdf)
            # sse = np.sum((hist-pdf) ** 2)
            rmse = np.sqrt(np.mean((hist-pdf)**2))
            if rmse < best_rmse:
                best_dist = dist_func
                best_params = params
                best_rmse = rmse
        except Exception as e:
            print(f"Error fitting {dist_name}: {str(e)}")

    print(f"Best fitting distribution: {best_dist.name}")
    print(f"Parameters: {best_params}")
    print(f"Best rmse: {best_rmse}")


    ### Find base threshold_0
    threshold_measured = np.percentile(anomaly_scores_no_extream_value, 99.9)  ## Percentile-based(CDF measurement-based) threshold

    cdf_x = np.linspace(min(filtered_anomaly_scores), max(filtered_anomaly_scores), bins)
    cdf_y = best_dist.cdf(cdf_x, *best_params)

    z_score = z_score_threshold;  # Set the z_score threshold based on normal PDF assumption
    # z_score = 3; 
    target_p_value = 1 - stats.norm.cdf(z_score)
    print(f"p-value for z-score {z_score}: {target_p_value:.20f}")
    
    #####
    threshold_estimated = best_dist.ppf(1-target_p_value, *best_params)
    
    # Linear combination of percentile-based threshold and best PDF-based threshold values
    # threshold_0 = 0.7*threshold_estimated + 0.3*threshold_measured
    threshold_0 = 0.9*threshold_estimated + 0.1*threshold_measured

    ### Find threshold_1 (Scale-up the threshold_0 accrording to data sparsity)
    count_zeros = np.sum(training_data == 0); count_100s = np.sum(training_data == 100);
    total_len = len(training_data); 
    ratio_of_zeros_100s = (count_zeros + count_100s) / total_len
    print('ratio_of_zeros_100s : ', ratio_of_zeros_100s)
    
    # count_zeros = (y_data_0 == 0).sum(); count_100s = (y_data_0 == 100).sum();
    # total_len = len(y_data_0); 
    # ratio_of_zeros_100s = (count_zeros + count_100s) / total_len
    # print('ratio_of_zeros_100s : ', ratio_of_zeros_100s)

    epsilon = np.finfo(float).eps
    # threshold_1 = threshold_0 * np.power(1/(1-ratio_of_zeros_100s+epsilon), 1.15) * 1
    threshold_1 = threshold_0 * np.power(1/(1-ratio_of_zeros_100s+epsilon), 0) * 1

    print("Threshold_measured : ", threshold_measured)
    print("Threshold_estimated : ", threshold_estimated)
    print("Threshold_0 : ", threshold_0)
    print("Threshold_1 : ", threshold_1, ' <-- ratio_of_zeros_100s : ', ratio_of_zeros_100s)

    ### Find threshold_2 (Application of final weight based on the importance of indicators)
    if re.search(r'ATTEMPT\(count\)|SUC\(count\)', col_name):
        attempt_factor = 1.3
        threshold_2 = threshold_1 * attempt_factor
        print(f"Threshold_2 (scaled-up by a factor of {attempt_factor} ) : ", threshold_2)
    # elif best_rmse > 0.01:
    #     threshold_2 = threshold_1 * 2
    else:
        threshold_2 = threshold_1 * 1.0
        print("Threshold_2 : ", threshold_2)\
        
        
    if fig_flag == 1:
        ########## Plot #############################
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax2 = ax1.twinx()
        ax1.hist(filtered_anomaly_scores, bins=bins, density=True, alpha=0.5, label='Histogram of Data')
        ax1.plot(cdf_x, best_dist.pdf(cdf_x, *best_params), 'r-', label='Estimated PDF')
        ax2.plot(cdf_x, cdf_y, 'g-', label='Estimated CDF')
        ax1.set_xlabel('Value'); ax1.set_ylabel('Density');
        ax2.set_ylabel('Cumulative Probability');
        ax1.legend(loc='upper left'); ax2.legend(loc='upper right')

        # ax1.axvline(threshold_1, color='k', linestyle='--', linewidth=1.5, label=f'p-value = {target_p_value:.4f}')
        # ax1.legend(loc='upper left')
        ax1.set_xlim(0, 0.001); ax1.set_ylim(0, 10000)

        plt.title(f"Histogram and Estimated {best_dist.name} Distribution")
        plt.show()
    
    return threshold_2, best_rmse


def find_local_z_score(data):
    
    local_z_score = np.zeros_like(data)

    mean = np.mean(data[0:len(data)-1])
    std_dev = np.std(data[0:len(data)-1])
    
    # if std_dev != 0:
    if std_dev >= 0.0001:
        local_z_score = (data[len(data)-1] - mean) / std_dev
    elif std_dev < 0.0001 and abs(data[len(data)-1] - mean) > 0.0001:
        local_z_score = 8 * abs(data[len(data)-1] - mean)
    else:
        local_z_score = 1
                
    return local_z_score


def score_post_processing(input_data, input_data_scaled, col_name, anomaly_score, threshold_2):

    input_data_scaled_diff = np.diff(input_data_scaled)
    # input_data_scaled_diff = np.insert(input_data_scaled_diff, 0, 0)

    local_z_score = np.abs(find_local_z_score(input_data_scaled_diff))

    local_z_score_threshold = 7.9*1

    if local_z_score > local_z_score_threshold:
        if "RATIO(%)" in col_name:
            anomaly_score_scaled = anomaly_score + ((local_z_score/8)*threshold_2)
        else:
            anomaly_score_scaled = anomaly_score * min(local_z_score*5, 300)
    else:
        anomaly_score_scaled = anomaly_score

    if "RATIO(%)" in col_name:
        # indices = (y_data_0 == 100)
        # y_data_3[indices] *= 0

        if (input_data[len(input_data)-1] > 99) & (input_data[len(input_data)-1] < 99.4):
            anomaly_score_scaled *= 0.5

        if (input_data[len(input_data)-1] >= 99.4):
            anomaly_score_scaled *= 0.03

        if (np.abs(input_data_scaled[len(input_data_scaled)-1]) > 3):
            anomaly_score_scaled *= np.abs(input_data_scaled[len(input_data_scaled)-1])*3

    elif "ERR(count)" in col_name:
        if (input_data[len(input_data)-1] < 10):
            anomaly_score_scaled *= 0.01       
 
    return anomaly_score_scaled
