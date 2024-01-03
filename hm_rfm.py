#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ciara Malamug, Jasmine Wong, Diana Xiao, Jane Zheng
DS2500
Professor Strange Lecture 1
Project #2: H&M Customer Segmentation (hm_rfm.py)
27 April 2022
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# create global constant for filename
HM = "hm_trans_clean.csv"

# matches the number of segments we have for RFM
K = 11

#=============================================================================

def find_recency(df, cur_time):
    '''
    when given a full dataframe of customer transactions, returns modified 
    dataframe with time since last transaction column for each customer

    Parameters
    ----------
    df : df with all customer transactions
    cur_time : datetime for current time (accounting period end date)

    Returns
    -------
    df : df with added time since last transaction column
    '''
    
    # group df by customer id to clean out repeat trans by same cust
    df = df.groupby(["customer_id"])

    # locates the most recent purchase date for each cust & reset index
    df = df.agg(max_date = ('t_dat', np.max)).reset_index()
    
    # calc last transaction timedelta (time since last purchase) for each cust
    df["last_trans_td"] = cur_time - df["max_date"]
    
    # make timedeltas into integer types
    df['last_trans_td'] = df['last_trans_td'].dt.days.astype('int16')
    
    return df

#=============================================================================

def find_freq(df):
    '''
    when given a full dataframe of customer transactions, returns modified 
    dataframe with number of visits column for each customer

    Parameters
    ----------
    df : df with all customer transactions

    Returns
    -------
    df : df with added num_visits column
    '''
    
    # group df by cust id and trans date to get items purchased by customer
    # on certain date
    df = df.groupby(["customer_id", "t_dat"])\
        .size().reset_index(name = "items_bought")
    
    # group df again by customer id to get total num of visits by each cust
    df = df.groupby(["customer_id"]).size().reset_index(name = "num_visits")

    return df

#=============================================================================

def find_monetary(df):
    '''
    when given a full dataframe of customer transactions, returns modified 
    dataframe with total exp for each customer

    Parameters
    ----------
    df : df with all customer transactions

    Returns
    -------
    df : df with tot_exp column as sum of all transactions
    '''
    
    # group df by customer id & find sum of price col for each customer
    df = df.groupby(["customer_id"])[["customer_id", "price"]].sum()
    
    # rename price column to be tot_exp since it now represents each cust's 
    # total expenditures
    df.rename(columns = {"price" : "tot_exp"}, inplace = True)
    df = df.reset_index()
    
    return df

#=============================================================================

def sort(df, col_name, order):
    '''
    Sort df values based on column name

    Parameters
    ----------
    df : df of interest
    col_name : name of column that we want to sort by
    order: boolean statement, True = ascending, False = descending
    
    Returns
    -------
    sorted_df : df with values sorted by col_name
    '''
    
    # sort df values by ascending
    sorted_df = df.sort_values(by = [col_name], ascending = order)
    
    return sorted_df

#=============================================================================

def make_n_strata(sorted_df, n):
    '''
    when given a sorted dataframe, make_n_strata splits sorted_df into n
    strata based on index and returns a list of dfs (each df = 1 strata)

    Parameters
    ----------
    sorted_df : sorted df that we want to stratify
    n : int representing the number of strata we want to divide the data into

    Returns
    -------
    strata_lst : list of dfs (each df = 1 strata based on percentile group)
    Ex: if n = 10 , 1st df is customers in the top 10%
    '''
    
    # create empty list to store strata
    strata_lst = []
    
    # calc length of each strata for the data
    split_idx = len(sorted_df) // n
    
    # set starting index at 0 and ending index as length of 1 strata
    start = 0
    end = split_idx
    
    # iterate through num of desired strata
    for split in range(n):
        
        # extract rows in strata's range & add df to strata_lst
        temp_df = sorted_df.iloc[start:end, :]
        strata_lst.append(temp_df)
        
        # update start and end positions for next strata
        start += split_idx
        end += split_idx
        
    return strata_lst

#=============================================================================

def stratified_sample(strata_lst, n):
    '''
    when given a list of strata dfs, extracts n random rows to make sample

    Parameters
    ----------
    df_lst : lst of strata dfs
    n : int representing the number of random rows we want from each strata

    Returns
    -------
    sample_lst : list of dfs containing randomly selected customers from each
    strata (length of list = num of strata)
    '''
    
    # create empty sample list
    sample_lst = []
    
    # iterate through each strata df
    for df in strata_lst:
        
        # select random customers from strata and add df to sample_lst
        random_custs = df.sample(n)
        sample_lst.append(random_custs)
        
    return sample_lst

#=============================================================================

def rank(df_lst, col_name):
    ''' 
    Take a list of df, lst index, and rank value.
    Make new df with new added rank column with corresponding rank 1-5, 5 
    being the highest.
    
    Parameters
    ----------
    df_lst: lst of df
    col_name: name of column (str)

    Returns
    -------
    df_empty: df with the new added column 
    '''
    
    # create empty df
    df_empty = pd.DataFrame()
    
    # iterate over the range of len of df_lst  
    for i in range(len(df_lst)):
        
        # for each df, add a new column with the values equal to 1 plus the 
        # index of the df and then append the df to empty df
        df = df_lst[i]
        df[col_name] = i+1
        df_empty = df_empty.append(df)
        
        # reset index of df
        df_empty.reset_index()
        
    return df_empty

#=============================================================================

def merge_df(rec_lst, freq_lst, mon_lst):
    """
    combine the df in each list of dfs into one df

    Parameters
    ----------
    rec_lst : list of df with customer and recency info
    freq_lst : list of df with customer and frequency info
    mon_lst : list of df with customer and monetary info 

    Returns
    -------
    all_df : df with all dfs in each list of dfs
    """
    
    # concat each list of dfs into dfs
    rec_df = pd.concat(rec_lst)
    freq_df = pd.concat(freq_lst)
    mon_df = pd.concat(mon_lst)
    
    # concat all list of dfs into one df
    all_df = pd.concat([rec_df, freq_df, mon_df])
    
    return all_df

#=============================================================================

def map_columns(df1, df2, df3, col_name):
    """
    Takes in 3 dfs, and the name of the column used as index.
    Combines 3 dfs into one df.

    Parameters
    ----------
    df1 : df with customer info and R_score
    df2 : df with customer info and F_score
    df3 : df with customer info and M_score
    col_name : name of column (str)

    Returns
    -------
    df1_copy : df with desired data from df1, df2, and df3
    """
    
    # make a copy of original df1
    df1_copy = df1.copy()
    
    # map over the F_score column from df2 to df1_copy 
    df1_copy["F_score"] = \
        df1_copy[col_name].map(df2.set_index(col_name)['F_score'])
        
    # map over the M_score column from df3 to df1_copy
    df1_copy["M_score"] = \
        df1_copy[col_name].map(df3.set_index(col_name)['M_score'])
    
    return df1_copy

#=============================================================================

def rfm_score(df, col1, col2, col3):
    '''
    Take in df, 3 column_names, make a rfm_score column.
    Return new_df with rfm_score.
    
    Parameters
    ----------
    df: df with customer info, R, F, and M scores. 
    col1: name of column (str)
    col2: name of column (str)
    col3: name of column (str)
    
    Returns
    -------
    df_copy: df with new added RFM_score column.
    '''
    
    # make a copy of the original df
    df_copy = df.copy()
    
    # make a new RFM_score column and set its value as the str addition of 
    # the 3 columns from the parameter
    df_copy["RFM_score"] = df_copy[col1].astype(str) + \
        df_copy[col2].astype(str) + df_copy[col3].astype(str)
        
    return df_copy

#=============================================================================             

def segment_cust(df, r_low_range, r_up_range, f_low_range, f_up_range, 
                 m_low_range, m_up_range):
    """
    when given a df and lower and upper ranges for the r, f, and m scores, 
    segment the customers into different groups based on the given ranges for 
    r, f, and m scores. 

    Parameters
    ----------
    df : df with customer info, R, F, and M scores. 
    r_low_range : lower range of R score (int)
    r_up_range : upper range of R score (int)
    f_low_range : lower range of F score (int)
    f_up_range : upper range of F score (int)
    m_low_range : lower range of M score (int)
    m_up_range : upper range of M score (int)

    Returns
    -------
    segment : df with segmented customer info
    """
    
    # .loc recency range based on rec range
    segment = df.loc[(df["R_score"] <= r_up_range) & \
                     (df["R_score"] >= r_low_range)]
    
    # from the segment based on .loc recency range, .loc freq based on 
    # freq range to further cut down the segment
    segment = segment.loc[(segment["F_score"] <= f_up_range) & \
                          (segment["F_score"] >= f_low_range)]
    
    # from the segment of .loc rec range + .loc freq range, .loc mon based on 
    # mon range to further cut down segment
    segment = segment.loc[(segment["M_score"] <= m_up_range) & \
                          (segment["M_score"] >= m_low_range)]
    
    return segment 
    
#=============================================================================   

def make_pie(df_lst, label_lst, color_lst, title):
    """
    when given a list of df, make a pie chart based on the df sizes, sets label
    and colors based on given parameters.

    Parameters
    ----------
    df_lst : a list of dataframes 
    color_lst : a list of colors
    label_lst : a list of pie chart labels
    title : title of the pie chart
    """
    
    # create an empty lst to store df sizes
    segment_size = []
    
    # iterate over each df in df_lst and store the row count in lst 
    for df in df_lst:
        count = df.shape[0]
        segment_size.append(count)
    
    # plot pie chart
    plt.figure(figsize = (8, 6))
    plt.pie(segment_size, labels = label_lst, colors = color_lst)
    plt.xticks(fontsize = 15)
    plt.title(title, fontsize = 20, fontweight = "bold")
   
    # save and show figure
    plt.savefig(title, dpi = 300)
    plt.show()

#=============================================================================   

if __name__ == "__main__":
    
    # Step 1: Read in the data

    # read the HM transaction csv into pandas df
    hm_df = pd.read_csv(HM)
    
    # turn the t_dat column (time data) into a datetime type
    hm_df["t_dat"] = pd.to_datetime(hm_df["t_dat"])
    
    # Step 2: Extract recency, frequency, monetary value data for each unique
    # customer id based on customer's transactions
    
    # make df w/ frequency (num of trans on diff dates) for each unique cust id
    freq_df = find_freq(hm_df)
    
    # make df w/ monetary (total expendature) of each unique cust id
    mon_df = find_monetary(hm_df)
    
    # set current time/most recent time to be the max date in the hm data
    cur_time = hm_df["t_dat"].max()
    
    # make df w/ recency (time since last transaction) for each unique cust id
    rec_df = find_recency(hm_df, cur_time)
    
    # Step 2b: make visualizations of r dist, f dist, m dist and all customers
    
    # create histogram of frequency distribution & format it
    sns.displot(freq_df["num_visits"], bins = 10)
    plt.yscale("log")
    plt.title("Frequency Distribution")
   
    plt.tight_layout()
    plt.subplots_adjust(top = 0.85)
    plt.savefig("Frequency Distribution.png")
    plt.show()
    
    # create histogram of monetary distribution & format it
    sns.displot(mon_df["tot_exp"], bins = 10)
    plt.yscale("log")
    plt.title("Monetary Distribution")
   
    plt.tight_layout()
    plt.subplots_adjust(top = 0.85)
    plt.savefig("Monetary Distribution.png")
    plt.show()
    
    # create histogram of recency distribution & format it
    sns.displot(rec_df["last_trans_td"], bins = 10)
    plt.xlabel("Days Since Last Visit")
    plt.title("Recency Distribution")
   
    plt.tight_layout()
    plt.subplots_adjust(top = 0.85)
    plt.savefig("Recency Distribution.png")
    plt.show()
    
    # create 3d plot of all customers based on rfm metrics & format it
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(freq_df["num_visits"], mon_df["tot_exp"], 
               rec_df["last_trans_td"])
    
    ax.set_xlabel("num_visits")
    ax.set_ylabel("tot_exp")
    ax.set_zlabel("last_trans_td")
    plt.title("All Customers By RFM Metrics")
    plt.savefig("3d All Cust", dpi = 300)
    plt.show()
    
    # Step 3: create consolidated df with each customer's rec, freq, mon data
    
    # make a copy of recency df to map freq and mon on
    rfm = rec_df.copy()
    
    # drop max_date column since it is no longer relevant to rfm
    rfm = rfm.drop("max_date", axis = 1)
    
    # map freq and mon to make consolidated df
    rfm["num_visits"] = \
        rfm["customer_id"].map(freq_df.set_index("customer_id")["num_visits"])
        
    rfm["tot_exp"] = rfm["customer_id"].map(mon_df.set_index("customer_id")\
                                         ["tot_exp"])

    # Step 4: Extract a stratified sample of customers to run rfm analysis on
    
    # create strata based on percentiles for recency, freq, and mon
    # sort dfs
    rec_sort = sort(rfm, "last_trans_td", True)
    freq_sort = sort(rfm, "num_visits", True)
    mon_sort = sort(rfm, "tot_exp", True)
    
    # split rec df into 10 strata by position and grab a sample of 20 from
    # each strata (equal representation from each percentile range)
    rec_strata = make_n_strata(rec_sort, 10)
    sample_by_rec = stratified_sample(rec_strata, 20)
  
    # do the same to frequency df to portray proportional range of freq
    freq_strata = make_n_strata(freq_sort, 10)
    sample_by_freq = stratified_sample(freq_strata, 20)

    # do the same to monetary df to portray proportional range of expenditures
    mon_strata = make_n_strata(mon_sort, 10)
    sample_by_mon = stratified_sample(mon_strata, 20)
    
    # merge 200 customers from each metric to get full sample of 600 customers
    sample_df = merge_df(sample_by_rec, sample_by_freq, sample_by_mon)    
    
    # check for duplicates and drop if there are any
    sample_df.drop_duplicates(inplace = True)
    
    # Step 5: Execute Marketing RFM Analysis
    
    # Step 5a: rank customers by assigning 5 discrete scores (1-5) for each
    # metric (Ex: 1 = worst, 5 = best)
    
    # similar to creating strata: sort, split & assign recency rank to sample
    num_ranks = 5
    
    # since lower number of days since last purchase is better, the sort order
    # of r is False = descending 
    sample_r_sort = sort(sample_df, "last_trans_td", False)
    sample_r_strata = make_n_strata(sample_r_sort, num_ranks)
    ranked_r = rank(sample_r_strata, "R_score")
    
    # assign freq rank to sample's customers
    sample_f_sort = sort(sample_df, "num_visits", True)
    sample_f_strata = make_n_strata(sample_f_sort, num_ranks)
    ranked_f = rank(sample_f_strata, "F_score")  
    
    # assign monetary rank to sample's customers
    sample_m_sort = sort(sample_df, "num_visits", True)
    sample_m_strata = make_n_strata(sample_m_sort, num_ranks)
    ranked_m = rank(sample_m_strata, "M_score")  
    
    # combine r, f, and m scores into rfm score
    merged_scores = \
        map_columns(ranked_r, ranked_f, ranked_m, "customer_id")
    
    sample_rfm_scored = rfm_score(merged_scores, "R_score", "F_score", 
                                    "M_score")
    
    # Step 5b: segment customers based on rfm ranges into 11 different based 
    # on marketing theory 
    
    champions = segment_cust(sample_rfm_scored, 4, 5, 4, 5, 4, 5)
    loyal_cust = segment_cust(sample_rfm_scored, 2, 5, 3, 5, 3, 5)
    pot_loy = segment_cust(sample_rfm_scored, 3, 5, 1, 3, 1, 3)
    rec_cust = segment_cust(sample_rfm_scored, 4, 5, 0, 1, 0, 1)
    prom = segment_cust(sample_rfm_scored, 3, 4, 0, 1, 0, 1)
    cust_need_att = segment_cust(sample_rfm_scored, 2, 3, 2, 3, 2, 3)
    ab_to_sleep = segment_cust(sample_rfm_scored, 2, 3, 0, 2, 0, 2)
    at_risk = segment_cust(sample_rfm_scored, 0, 2, 2, 5, 2, 5)
    cant_lose = segment_cust(sample_rfm_scored, 0, 1, 4, 5, 4, 5)
    hiber = segment_cust(sample_rfm_scored, 1, 2, 1, 2, 1, 2)
    lost = segment_cust(sample_rfm_scored, 0, 2, 0, 2, 0, 2)
    
    # Step 5c: create visualizations for Marketing RFM Analysis Method
    
    # create list of segments, labels & colors to help with plotting
    segment_lst = [champions, loyal_cust, rec_cust, pot_loy, prom, 
                   cust_need_att, ab_to_sleep, at_risk, cant_lose, hiber, lost]
    
    seg_label_lst = ["Champions", "Loyal Customers", "Recent Customers", 
                     "Potential Loyalist", "Promising", 
                     "Customers Needing Attention", "About to Sleep", 
                     "At Risk", "Can't Lose Them", "Hibernating", "Lost"]
    
    seg_color = ["royalblue", "mediumturquoise", "darkgreen", "lightgreen",
                 "gold", "tomato", "mediumpurple", "palevioletred", "red", 
                 "sienna", "black"]
    
    # make pie chart of segment sizes
    make_pie(segment_lst, seg_label_lst, seg_color, "RFM Segment Sizes")
  
    # create 3d plot of 600 sample customers with dimensions: R,F,M
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    
    # establishes what will be the x, y, and z for the 2 plots
    x = "num_visits"
    y = "tot_exp"
    z = "last_trans_td"
    
    # iterate through the segments
    for i in range(len(segment_lst)):
        # plots the x, y, and z for each segment, using the same labels and 
        # colors as the pie chart
        ax.scatter(segment_lst[i][x], segment_lst[i][y], segment_lst[i][z], 
                   color = seg_color[i], label = seg_label_lst[i])
    
    # sets lables and title
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title("RFM 3d Plot Segmentation")
    plt.savefig("RFM_3d_segment.png", dpi = 300)
    
    plt.show()
    
    # Step 6: Implement K-means clustering to segment the sample of 600 cust
    
    # Step 6a : Run k-mean clustering
    # split into the features we will use to cluster
    X = sample_df[['last_trans_td', 'num_visits', 'tot_exp']]
    
    # create and fit the model
    km = KMeans(n_clusters = K)
    km.fit(X)
    
    # print the intertia
    print("Inertia:", km.inertia_)
    
    # add the group columns to the df
    sample_df["group"] = km.labels_
    
    # split into the different groups
    gp_lst = []
    gp_label = sample_df["group"].unique()
    gp_label.sort()
    for i in gp_label:
        gp_lst.append(sample_df[sample_df["group"] == i])
        
    # creates a plot 
    fig = plt.figure() 
    ax = fig.add_subplot(111, projection = '3d')
    
    # goes through the group list and plots each group in different colors
    for i in range(len(gp_lst)):
        ax.scatter(gp_lst[i][x], gp_lst[i][y], gp_lst[i][z], 
                   color=seg_color[i])
    
    # sets labels and titles
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)
    ax.set_title("RFM 3d Plot Clustering")
    
    plt.savefig("RFM_3d_clust.png", dpi = 300)
    
    plt.show()
    
    # make pie chart of clustering segment sizes
    make_pie(gp_lst, gp_label, seg_color, "Clustering Segment Sizes")
