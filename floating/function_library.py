#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 15:48:08 2023

@author: hadiwidjajastella
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import pvlib
import pickle
import numpy as np
import seaborn as sns
import scipy.stats as stats


# =============================================================================
# Floating PV Analysis : Functions
# =============================================================================

# The flow is as follows : 

# Reorganisation -> Cleaning -> Imputing -> Performance Metrics -> PLRs


# =============================================================================
# =============================================================================
# =============================================================================
# # #  ## Reorganisation ##
# =============================================================================
# =============================================================================
# =============================================================================

# The data of each sensor station is stored in daily .txt files . 
# The daily files first need to be merged temporally across the timescope of the study. 

def merge_index_temporal(station_number, fp):
    """
    Parameters
    ----------
    station_number : int
        The station number of the sensors.
        e.g. 135
    fp : string
        The directory of the .txt files containing daily data.
    fp_save : string
        The directory where the merged data in the form of a csv file will be stored.

    Returns
    -------
    df_merged : pandas dataframe
    Dataframe containing the data merged temporally across the duration of start date to end date

    """
    # Get to the root directory 
    root_dir = fp+'['+str(station_number)+']/Converted to TXT'

    
    # Initialize an empty list to store the file paths
    file_paths = []
    
    # Iterate over the yearly subfolders
    
    for year_folder in os.listdir(root_dir):
        year_path = os.path.join(root_dir, year_folder)
        
        # Check if the item in the year folder is a directory
        if not os.path.isdir(year_path):
            continue
        
        # Iterate over the monthly subfolders
        for month_folder in os.listdir(year_path):
            month_path = os.path.join(year_path, month_folder)
            
            # Check if the item in the month folder is a directory
            if not os.path.isdir(month_path):
                continue
            
            # Iterate over the text files
            for file_name in os.listdir(month_path):
                file_path = os.path.join(month_path, file_name)
                
                # Check if the item is a file
                if os.path.isfile(file_path):
                    file_paths.append(file_path)
    
    # Initialize an empty list to store the dataframes
    dfs = []
    
    # Read each text file as a dataframe and append it to the list
    for file_path in file_paths:
        df = pd.read_csv(file_path, sep='\t')  # Specify the tab separator
        dfs.append(df)
    
    # Concatenate all the dataframes into a single dataframe
    df_merged = pd.concat(dfs)
    
    #change the timestamp to datetime
    df_merged['Tm'] = pd.to_datetime(df_merged['Tm'])
    
    #set the index to time
    df_merged = df_merged.set_index('Tm') 
    
    #sort in chronological order
    df_merged = df_merged.sort_index()  

    return df_merged

# =============================================================================
# # Test merge_index_temporal
# station_number = 135
# fp = '/Users/hadiwidjajastella/Documents/MEng/floating/version_2_11-6/og_data/'
# start_date = '2017-01-01'
# end_date = '2022-04-30'
# =============================================================================

def create_timestamps(start_date,end_date,latitude= 1.349578,longitude=103.639491 ,timezone='Asia/Singapore',freq='T',solar_angle_cutoff = 15):
    """

    Parameters
    ----------
    start_date : string
        The starting date in the form 'YYYY-MM-DD', before which sensor data will not be included.
        e.g. '2017-01-01’
    end_date : string
        The ending date in the form 'YYYY-MM-DD', after which sensor data will not be included.
    latitude : float
        latitude of location of PV installation.
    longitude : float
        longitude of location of PV installation.
    timezone : string, optional
        Timezone of minutes used. The default is 'Asia/Singapore'.
    freq : string, optional
        Frequency of time data, which is minutes in the case of Tengeh. The default is 'T'.
    solar_angle_cutoff : int, optional
        The angle of the sun below which the data is cutoff. The default is 15.

    Returns
    -------
    sun_minutes : series
        a time index to be used to order the sensor data.

    """
    # Create a date range for the year 2017 with a minute frequency
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq, tz=timezone)
    
    # Get solar position for the specified location and date range
    solar_position = pvlib.solarposition.get_solarposition(date_range, latitude, longitude)
    
    # Calculate solar elevation angle
    solar_elevation = solar_position['elevation']
    
    # Find the minutes when the sun makes an x degree angle to the horizon at sunrise and sunset
    # at low elevations, there are problems with the data in the irradiance sensors, thus sunrise & sunset is usually cutoff
    sun_minutes = date_range[(solar_elevation > solar_angle_cutoff) & (solar_elevation < 180-solar_angle_cutoff)]
    sun_minutes = sun_minutes.tz_localize(None)
    return sun_minutes


# =============================================================================
# # name of the data file that has been night-filtered and temporally merged
# csv_suffix = 'day_merged.csv'
# 
# #directory of folder to save the csv
# fp_save = '/Users/hadiwidjajastella/Documents/MEng/floating/version_4/'+str(station_number)+csv_suffix
# 
# df_merged = df_merged.reindex(sun_minutes)
# df_merged.to_csv(fp_save)
# =============================================================================


def generate_sensor_csv(sensor_list,fp_load, sensor_dict,sun_minutes,csv_suffix = '_day_merged.csv'):
    """   

    Parameters
    ----------
    sensor_list : list of strings
        A list of sensors to be included in the data frame.
    fp_load : string
        Directory of where the station data is stored, already with merged temporally with filtered timestamps.
    csv_suffix : string
        the suffix of the csv file containing merged, time-indexed station data
    sensor_dict : dict
        A dictionary mapping each sensor to the station they are in. Stored in a .pickle file which is loaded into the program.
    
    Returns
    -------
    df_sensor_list : dataframe
        A dataframe containing all the data from sensors in sensor_list, where each sensor is a column.

    """
    
    dfs = []
    
    sensor_sta = {key: sensor_dict[key] for key in sensor_list}

    #concatenate relevant columns into the df list
    for sensor, station in sensor_sta.items():
        fp = fp_load+str(station)+csv_suffix
        df = pd.read_csv(fp,)
        if sensor in df.columns:
            df = df[[sensor]]
            dfs.append(df) 
        
    df = pd.concat(dfs, axis=1)
    df = df.set_index(sun_minutes)
    
    return df



def filter_range(df,value_range, columns=None):
    """
    Filter values outside a given range in specified columns of a DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame to filter.
    - columns (list of str, default None): The columns in which to apply the filter. If None, it will default to all columns.
    - value_range (list of 2 floats): The minimum and maximum acceptable values for the filter.

    Returns:
    - DataFrame: A DataFrame with values outside the given range replaced with NaN in the specified columns.
    """

    # If no columns are specified, default to all columns of the DataFrame
    if columns is None:
        columns = df.columns

    df_filtered = df.copy()

    for col in columns:
        mask = ~df[col].between(value_range[0], value_range[1])  # Create a mask for values outside the range
        df_filtered.loc[mask, col] = np.nan  # Replace values outside the range with NaN

    return df_filtered



def filter_iqr(df):
    """
    For each column in a DataFrame, filter out values outside of the 
    interquartile range (IQR) and replace them with NaN.
    """
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    filtered_df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))]
    return filtered_df


def moving_average(data, N):
    return np.convolve(data, np.ones((N,))/N, mode='valid')


def visualize_effect(original_df, filtered_data, title, label):
    plt.figure(figsize=(10,6))
    plt.plot(original_df.index, original_df.iloc[:, 0], label='Original Data', color='blue')
    plt.plot(original_df.index[:len(filtered_data)], filtered_data, label=label, color='red', linestyle='dashed')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


# Example Usage:
# df_filtered = apply_range_filter(df, columns=['col1', 'col2'], value_range=[10, 50])
# =============================================================================
# 
# def filter_range(df,range_dict=range_dict):
#     """
#     
#     Parameters
#     ----------
#     range_dict : dict
#         A dictionary with the ranges for key sensor parameters.
#         e.g. range_dict = {'Tamb':[10,50],'Hamb':[0,100],'Wind':[0,30]}
# 
#     df : dataframe
#         dataframe to apply the filter to.
# 
#     Returns
#     -------
#     df_filtered : dataframe
#         filtered dataframe.
# 
#     """
# 
#     df_filtered = df.copy()  # Create a copy of the DataFrame to keep the original unaltered
#     for sensor, value_range in range_dict.items():
#         for col in df.columns:
#             if sensor in col:
#                 mask = ~df[col].between(value_range[0], value_range[1])  # Mask values outside the range
#                 df_filtered.loc[mask, col] = np.nan
# 
#     return df_filtered
# 
# =============================================================================

# =============================================================================
# def filter_abnormal_bins(df, n = 100):
#     """
#     
#     Parameters
#     ----------
#     n : omt
#         number of bins to consider
#         # n should be very large to avoid deleting good data
# 
# 
#     df : dataframe
#         dataframe to apply the filter to.
# 
#     Returns
#     -------
#     df_filtered : dataframe
#         filtered dataframe without data in bins that are too high .
# 
#     """
#     # Create an empty DataFrame to store the modified data
#     df_filtered = df.copy()
#     
#     # Iterate over each column in the DataFrame
#     for column in df_filtered.columns:
#         
#         # Resample the column to hourly intervals and calculate the mean
#         hourly_means = df_filtered[column].resample('H').mean()
#         
#         # Create bins for the hourly mean values
#         bins = pd.cut(hourly_means, bins=n, labels=False)
#         
#         # Calculate the bin counts
#         bin_counts = pd.value_counts(bins)
#         
#         # Find the highest and next highest bin counts
#         highest_count = bin_counts.max()
#         
#         sorted_counts = bin_counts.sort_values(ascending=False)
#         
#         if len(sorted_counts) > 1:
#             next_highest_count = sorted_counts.iloc[1]
#         else:
#             # Handle the situation where there's only one count or it's empty
#             # Depending on the logic of your function, you might set next_highest_count to some default value or handle it differently
#             next_highest_count = 0  # or another appropriate value or action
#             
#             
#         # Check if the highest count is 1.5x higher than the next highest count
#         if highest_count > 1.5 * next_highest_count:
#             print('abnormal bin detected')
#             # Identify the bin with the highest count
#             highest_bin = bin_counts.idxmax()
#         
#             # Create a mask for hours corresponding to the highest bin
#             mask = hourly_means.index.floor('H').isin(hourly_means[bins == highest_bin].index.floor('H'))
#             hourly_outlier = hourly_means[mask].index.values
#             
#                 
#             # Loop through each hour in the 'hour_list'
#             for hour in hourly_outlier:
#                 # Define the start and end time range for each hour
#                 start_time = hour
#                 end_time = hour + pd.DateOffset(hours=1) - pd.DateOffset(minutes=1)
#                 
#                 # Select the data within the time range and set it to NaN
#                 df_filtered.loc[start_time:end_time, column] = np.nan
# 
#     return df_filtered
# =============================================================================


def filter_by_nan_percentage(df, freq='T',threshold=0.9):
    """
    Filter out days in the dataframe where a given percentage of data is NaN.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with minutely data and a datetime index.
    threshold : float, optional
        The percentage (0 to 1) of NaN values above which the entire day's data is set to NaN.
        Default is 0.8 (i.e., 80%).
        
    Returns
    -------
    DataFrame
        DataFrame with days filled with NaN where NaN values exceed the given threshold.
    """

    for col in df.columns:
        # Calculate the percentage of NaN values for each day for the column
        nan_percentage = df[col].resample(freq).apply(lambda x: x.isna().mean())
        
        # Identify dates where the percentage of NaN values for the column exceeds the threshold and is less than 100%
        dates_to_nullify = nan_percentage[(nan_percentage > threshold) & (nan_percentage < 1)].index

        # For each of those dates, set the data values for the column to NaN
        for date in dates_to_nullify:
            mask = (df.index.date == date.date())
            df.loc[mask, col] = np.nan

    return df


def filter_combined(df):
    df_filtered = filter_range(df)
    df_filtered = filter_abnormal_bins(df_filtered, n = 100)
    df_filtered = filter_by_nan_percentage(df_filtered, threshold=0.9)
    return df_filtered






def filter_abnormal_bins(df, n =200):
    """Filter out hours where the count of a particular bin is 1.5x greater than the next highest bin count."""

    # Create an empty DataFrame to store the modified data
    df_filtered = df.copy()
    
    # Iterate over each column in the DataFrame
    for column in df_filtered.columns:
        
        deleted_bin_count = 0
        abnormal_detected = True

        # Continue until the condition is not satisfied
        while abnormal_detected:
            
            # Resample the column to hourly intervals and calculate the mean
            hourly_means = df_filtered[column].resample('H').mean()
            
            # Create bins for the hourly mean values
            bins = pd.cut(hourly_means, bins=n, labels=False)
            
            # Calculate the bin counts
            bin_counts = pd.value_counts(bins)
            
            # Find the highest and next highest bin counts
            highest_count = bin_counts.max()
            next_highest_count = bin_counts.sort_values(ascending=False).iloc[1]

            # Check if the highest count is 1.5x higher than the next highest count
            if highest_count > 1.5 * next_highest_count:
                print('Abnormal bin detected in column:', column)
                # Identify the bin with the highest count
                highest_bin = bin_counts.idxmax()

                # Create a mask for hours corresponding to the highest bin
                mask = hourly_means.index.floor('H').isin(hourly_means[bins == highest_bin].index.floor('H'))
                hourly_outlier = hourly_means[mask].index.values

                # Loop through each hour in the 'hourly_outlier' list
                for hour in hourly_outlier:
                    # Define the start and end time range for each hour
                    start_time = hour
                    end_time = hour + pd.DateOffset(hours=1) - pd.DateOffset(minutes=1)

                    # Select the data within the time range and set it to NaN
                    df_filtered.loc[start_time:end_time, column] = np.nan
                
                deleted_bin_count+=1
                
            else:
                abnormal_detected = False
        print('Number of Deleted Bins:'+str(deleted_bin_count))    
        
    return df_filtered







# =============================================================================
# =============================================================================
# =============================================================================
# # # Visualise
# =============================================================================
# =============================================================================
# =============================================================================

def scatter_plot_and_save(df_system, fig_fp,resample_freq='D',fig_name=None):
    """
    Plots and saves the data from df_system after resampling.

    Parameters:
    - df_system (pd.DataFrame): The dataframe to plot. Default is df_system_Pdc.
    - resample_freq (str): The frequency for resampling. Default is 'D' (daily).
    - fig_fp (str): File path for saving the plot. 
    - system (str): System name to be used in the filename.
    - sensor_type (str): Sensor type to be used in the filename.

    Returns:
    None. Displays the plot and saves it as a PNG file.
    """

    # Resample the data
    resampled_data = df_system.resample(resample_freq).mean()

    # Determine the number of columns in the DataFrame to set the number of subplots
    num_columns = len(resampled_data.columns)
    fig, axes = plt.subplots(num_columns, 1, figsize=(15, 6*num_columns))

    # Iterate through each column and plot it
    for i, column in enumerate(resampled_data.columns):
        sns.scatterplot(x=resampled_data.index, y=resampled_data[column], alpha=0.6, marker='.', ax=axes[i])
        axes[i].set_xlabel('Time')
        axes[i].set_ylabel(column)
        #axes[i].set_title(column)

    plt.tight_layout()
    plt.savefig(fig_name)
    
    if fig_name:
        plt.savefig(fig_name)
        
    plt.show()

# Example usage:
# plot_and_save(df_system=some_data, resample_freq='D', fig_fp='/path/to/save/', system='System1', sensor_type='TypeA')


def plot_nan_percentage_heatmap(df,freq = 'M', fig_name=None):
    """
    Creates a heatmap that shows the percentage of NaN values per month in the given DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame

    Returns:
    None
    """

    # Compute the percentage of NaNs per month
    nan_percentage = df.resample(freq).apply(lambda x: x.isna().mean() * 100)

    # Plot heatmap
    plt.figure(figsize=(16, 8))
    ax = sns.heatmap(nan_percentage.T, cmap="viridis", cbar_kws={'label': 'Percentage of NaNs'})

    # Format x-tick labels
    ax.set_xticklabels([pd.to_datetime(tick.get_text()).strftime('%b \'%y') for tick in ax.get_xticklabels()])

    # Set titles and labels
    plt.title('% of NaN Data ')
    #plt.ylabel()
    plt.xlabel('Time')
    plt.tight_layout()
    if fig_name:
        plt.savefig(fig_name)
    plt.show()



def compare_dfs(df, df_filtered, freq='D',fig_name=None):
    """
    Compares the effects of a filter on two DataFrames, df and df_filtered, by plotting scatterplots for each column.

    Parameters:
    - df (DataFrame): The original DataFrame.
    - df_filtered (DataFrame): The DataFrame after applying a filter.
    - freq (str, optional): Frequency for resampling. Default is 'H' for hourly.

    Returns:
    None
    """

    # Compute the means for each DataFrame based on the provided frequency
    means = df.resample(freq).mean()
    means_filtered = df_filtered.resample(freq).mean()

    # For each column, create subplots showcasing the effects of the filter
    for col in df.columns:
        fig, axes = plt.subplots(2, 1, figsize=(10, 15), sharex=True)
        
        # Plot for the original DataFrame
        sns.scatterplot(x=means.index, y=means[col], ax=axes[0], label="Original")
        axes[0].set_title(f"Original: {col}")
        axes[0].legend()
        
        # Plot for the filtered DataFrame
        sns.scatterplot(x=means_filtered.index, y=means_filtered[col], ax=axes[1], label="After filter", color="orange")
        axes[1].set_title(f"After filter: {col}")
        axes[1].legend()
        
        # Adjust the layout and show the plot
        plt.tight_layout()

        plt.show()



def compare_histograms(df, df_filtered, bins=100,fig_name=None):
    """
    Compares the distributions of two DataFrames, df and df_filtered, by plotting histograms for each column in a grid of subplots.

    Parameters:
    - df (DataFrame): The original DataFrame.
    - df_filtered (DataFrame): The DataFrame after applying a filter.
    - bins (int, optional): Number of bins for the histogram. Default is 100.

    Returns:
    None
    """
    
    # Set style for seaborn to make the plot look nice
    sns.set_style("whitegrid")

    # Number of columns to plot
    num_columns = len(df.columns)
    
    # Set up the grid of subplots
    fig, axes = plt.subplots(num_columns, 1, figsize=(10, 6*num_columns))
    
    # Ensure axes is iterable
    if num_columns == 1:
        axes = [axes]

    # For each column, create a subplot showcasing the effects of the filter
    for i, col in enumerate(df.columns):
        
        # Histogram for the original DataFrame
        sns.histplot(df[col], bins=bins, kde=True, label="Raw", alpha=0.5, color="blue", ax=axes[i])
        
        # Histogram for the filtered DataFrame
        sns.histplot(df_filtered[col], bins=bins, kde=True, label="After filters", alpha=0.5, color="orange", ax=axes[i])

        # Set titles, legends, and labels
        # axes[i].set_title(f"Comparison of {col} Distribution")
        axes[i].legend()
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    if fig_name:
        plt.savefig(fig_name)
        
    # Adjust layout
    plt.tight_layout()
    plt.show()
    
    

# Example usage
# compare_histograms(df_original, df_filtered)


# =============================================================================
# =============================================================================
# =============================================================================
# # # Help Functions
# =============================================================================
# =============================================================================
# =============================================================================

def filter_sensors_by_terms(sensor_list, study_terms):
    """
    Filters the sensor_list to return only those strings that contain all the strings in study_terms.

    Parameters:
    - sensor_list (list of str): List of sensor strings.
    - study_terms (list of str): List of study terms to filter by.

    Returns:
    - List of strings from sensor_list that contain all the study_terms.
    """

    filtered_sensors = [sensor for sensor in sensor_list if all(term in sensor for term in study_terms)]
    return filtered_sensors

# Function to find the longest consecutive missing data for each column
def longest_consecutive_missing(df):
    max_missing = {}
    for column in df.columns:
        missing = df[column].isnull().astype(int)
        consec_counts = missing.groupby((missing != missing.shift()).cumsum()).cumsum()

        if consec_counts.max() > 0:
            max_missing[column] = consec_counts.max()

    return max_missing


def plot_relative_difference_histogram(df, reference_column, n_bins=20, num_cols=3):
    # Get a list of all other sensor columns
    sensor_columns = [col for col in df.columns if col != reference_column]

    # Set up subplots
    num_subplots = len(sensor_columns)
    num_rows = -(-num_subplots // num_cols)  # Calculate the number of rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    fig.suptitle(f'Relative Difference to {reference_column}', y=1.02, fontsize=16)

    # Flatten the axes for easier iteration
    axes = axes.flatten()

    # Iterate through each sensor column and plot the relative difference
    for i, sensor_column in enumerate(sensor_columns):
        relative_difference = (df[sensor_column] - df[reference_column]) / df[reference_column]

        # Plot the relative difference histogram with specified number of bins
        sns.histplot(relative_difference, kde=True, bins=n_bins, ax=axes[i])
        axes[i].set_title(f'{sensor_column} vs {reference_column}')
        axes[i].set_xlabel('Relative Difference')
        axes[i].set_ylabel('Density')

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()
    
    
def longest_consecutive_missing(df):
    max_missing = {}
    for column in df.columns:
        missing = df[column].isnull().astype(int)
        consec_counts = missing.groupby((missing != missing.shift()).cumsum()).cumsum()

        if consec_counts.max() > 0:
            max_missing[column] = consec_counts.max()

    return max_missing

def calculate_longest_gap_grade(df):
    # Calculate the longest consecutive missing data for each column in days
    longest_gap = {column: int(round(value / (24 * 60))) for column, value in longest_consecutive_missing(df).items()}

    # Create a DataFrame with the results
    result_df = pd.DataFrame({'Longest gap (days)': longest_gap.values()}, index=df.columns)

    # Assign grades
    grade = pd.cut(result_df['Longest gap (days)'],bins=[-float('inf'), 15, 30, 90, float('inf')],labels=['A', 'B', 'C', 'D'])
    
    # Return both the numerical value and the result DataFrame with grades
    return longest_gap.values(), grade.values

def calculate_outliers_grade(df):
    # Calculate IQR for each column
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # Find outliers
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))

    # Calculate percentage of outliers for each column
    outliers_percentage = (outliers.mean() * 100).round(2)

    # Assign grades based on the percentage of outliers
    grade = pd.cut(outliers_percentage, bins=[-float('inf'), 10, 20, 30, float('inf')],
                   labels=['A', 'B', 'C', 'D'])

    # Return both the numerical value and the result DataFrame with grades
    return outliers_percentage.values, grade.values

def calculate_missing_percentage_grade(df):
    missing_percentage = (df.isnull().mean() * 100).round(2)
    
    # Assign grades based on missing percentage
    grade = pd.cut(missing_percentage, bins=[-float('inf'), 10, 25, 40, float('inf')],
                   labels=['A', 'B', 'C', 'D'])
    
    return missing_percentage, grade


def calculate_time_difference_grade(df):
    # Find the time difference between the first and last non-NaN values for each column
    time_difference = df.apply(lambda col: col.last_valid_index() - col.first_valid_index())
    time_difference = time_difference.dt.days
    # Assign grades based on the time difference
    grade_time_difference = pd.cut(time_difference, bins=[-float('inf'),365*2, float('inf')],
                                   labels=['F', 'P'])

    # Return both the numerical value and the result DataFrame with grades
    return time_difference.values, grade_time_difference.values


def calculate_anomalies_grade(df,range_values):
    # Count the number of non-NaN values outside the range of 100 to 1200 for each column
    anomalies_count = ((df < range_values[0]) | (df > range_values[1])).sum()

    # Calculate percentage of anomalies for each column
    anomalies_percentage = (anomalies_count / len(df) * 100).round(2)

    # Assign grades based on the percentage of anomalies
    grade_anomalies = pd.cut(anomalies_percentage, bins=[-float('inf'), 10, 25, 40, float('inf')],
                             labels=['A', 'B', 'C', 'D'])

    # Return both the numerical value and the result DataFrame with grades
    return anomalies_percentage.values, grade_anomalies.values

def calculate_uPLR(u2_a, u2_b, a, b):
    uPLR = np.sqrt((u2_a * ((12 / b) ** 2)) + (b * u2_b * (12 * a / (b ** 2) ** 2)))
    return uPLR

def calculate_confidence_interval(u2_a, u2_b, a, b, PLR, confidence_level=0.95):
    uPLR = np.sqrt((12 * b)**2 * u2_a + (b * (12 * a * b)**2) * u2_b)
    critical_value = stats.t.ppf((1 + confidence_level) / 2, df=1)
    MOE = critical_value * uPLR
    lower_bound = PLR - MOE
    upper_bound = PLR + MOE
    bounds = [lower_bound, upper_bound]
    return bounds
# =============================================================================
# =============================================================================
# =============================================================================
# Test
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# 
# from matplotlib.dates import HourLocator, DateFormatter
# 
# start_date = '2017-01-01'
# end_date = '2023-04-30'
# sun_minutes = create_timestamps(start_date,end_date,latitude= 1.349578,longitude=103.639491 ,timezone='Asia/Singapore',freq='T',solar_angle_cutoff = 15)
# fp_load = '/Users/hadiwidjajastella/Documents/MEng/floating/version_3/merged/'
# sensor_list = ['AvgPdc4s1_SSACl','AvgPdc4s2_SSACl','AvgPdc4s3_SSACl','AvgPdc4s4_SSACl']
# 
# df = generate_sensor_csv(sensor_list,fp_load, sensor_dict,sun_minutes,csv_suffix = '_day_merged.csv')
# df_filtered = filter_combined(df)
# 
# 
# plot_nan_percentage_heatmap(df_filtered)
# compare_dfs(df, df_filtered, freq='D')
# compare_histograms(df, df_filtered, bins=50)
# 
# 
# 
# 
# 
# test_data = df['AvgPdc4s1_SSACl']
# 
# # Plotting
# plt.figure(figsize=(15, 6))
# sns.scatterplot(x=test_data.index, y=test_data, alpha=0.6,marker='.')
# plt.xlabel('Time')
# plt.ylabel('Average DC Power Output of SSACl String 1 ')
# plt.tight_layout()
# plt.show()
# 
# 
# semi_constant_data = test_data[test_data.index.date == pd.Timestamp('2022-03-20').date()]
# 
# 
# # Plotting
# plt.figure(figsize=( 6,4))
# sns.scatterplot(x=semi_constant_data.index, y=semi_constant_data, alpha=0.6, marker='.')
# 
# 
# # Adjusting the x-axis to display only hours
# ax = plt.gca()
# ax.xaxis.set_major_locator(HourLocator())
# ax.xaxis.set_major_formatter(DateFormatter('%H'))
# 
# plt.ylim(937, 938)  # Setting y-axis limits
# plt.xlabel('Time on 2022-03-20')
# plt.ylabel('Average DC Power Output of SSACl String 1')
# plt.tight_layout()
# plt.show()
# =============================================================================
