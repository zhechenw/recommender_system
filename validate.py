# -*- coding: utf-8 -*-
"""
Created on Tue May 14 21:10:06 2019

@author: septe

this code validate the models using multiple metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_dataset, find_intersection


def load_test(test_path='./dataset/yahoo_r2/test_0.txt'):
    header = ['user_id', 'song_id', 'rating']
    test_df = pd.read_csv(test_path, sep='\t', names=header)
    test_df = test_df.set_index('user_id')
    return test_df


def load_stat(path='./raw_feature/train_0/new_train_0_user_info.csv'):
    stat_df = pd.read_csv('./raw_feature/train_0/train_0_user_info.csv', sep=',')
    stat_df.columns = ['user_id', 'mean', 'std']
    stat_df = stat_df.set_index('user_id')
    return stat_df
    
    
def get_true(user, test_df, train_df, stat_df):
    
    # extract user raw test data
    test_info = test_df.loc[user]
    true = list(test_info['rating'])
    test_song_list = list(test_info['song_id'])
    
    # extract user mean and std for standardization
    mean, std = list(stat_df.loc[user])
    
    # standardize raw test data
    user_rt = list(train_df.set_index('user_id').loc[user]['rating'])
    user_rt = (np.array(user_rt) - mean) / std
    
    # turn scores into ratings
    bin1 = np.histogram_bin_edges(user_rt, bins=5)
    true = (np.array(true) - mean) / std
    true = np.where(true<bin1[1], 1, 
                np.where(true<bin1[2], 2, 
                         np.where(true<bin1[3], 3, 
                                  np.where(true<bin1[4], 4, 5))))

    return true, test_song_list


def get_pred(user, k, test_song_list, train_set=0, function='correlation'):
    
    # load model
    model_path = './models/train_{}/{}/{}/{}_k_{}.csv' \
        .format(train_set, str(user).zfill(7), function, str(user).zfill(7), k)
    md = pd.read_csv(model_path, sep=',', names=['song_id', 'score'], 
                     index_col=['song_id'])
    md_song_list = list(md.index)
    md_song_score = list(md['score'])
    avg = md['score'].mean()
    pred = []
    
    # assign score to items in test dataset
    for song in test_song_list:
        if song in md_song_list:
            pred.append(md.loc[song]['score'])
        else:
            pred.append(avg)
            
    # turn scores into ratings
    bin2 = np.histogram_bin_edges(md_song_score, bins=5)
    pred = np.where(pred<bin2[1], 1, 
                np.where(pred<bin2[2], 2, 
                         np.where(pred<bin2[3], 3, 
                                  np.where(pred<bin2[4], 4, 5))))
    
    return pred, md_song_list


def get_error_conf(true, pred, test_song_list):

    conf_mx = confusion_matrix(true, pred)

    error = 0
    for i in range(len(test_song_list)):
        if true[i] != pred[i]:
            error += 1
    mean_error = error / len(test_song_list)

    return mean_error, conf_mx


def get_user_rating(target_user, df):
    return list(df.loc[target_user]['rating'])


def get_ame(user, k, test_df, train_df, plot=False, binary=False, 
                threshold=3, function='correlation', train_set=0):
    
    """
    this code calculate the absolute mean error (AME) for one user for rating prediction
    
    if binary, a threshold must been given to divide item into like and dislike
    """
    
    # generate ground truth and prediction
    true, test_song_list = get_true(user, test_df, train_df, stat_df)
    pred, md_song_list = get_pred(user, k, test_song_list, 
                                    train_set=train_set, function=function)
    
    if binary:
        true = np.where(true>threshold, 1, 0)
        pred = np.where(pred>threshold, 1, 0)
        mean_error = sum(np.where(true==pred, 0, 1))/len(test_song_list)
        title = 'true vs pred, user {}, k = {}, threshold = {}'.format\
                                            (user, k, threshold)
    else:
        title = 'true vs pred, user {}, k = {}'.format(user, k)
        mean_error = sum(abs(np.array(pred) - np.array(true))) / len(true)
    
    if plot:
        plt.plot(list(range(len(true))), true, c='b', marker='x', markersize=12)
        plt.plot(list(range(len(pred))), pred, c='r', marker='o', markersize=10)
        plt.legend(['true', 'pred'])
        plt.title(title)
        plt.ylabel('rating')
        plt.xlabel('song')
        plt.xticks(list(range(1,11)))
        plt.show()

    return true, pred, mean_error


def find_best_k(user_list, k_range, test_df, train_df, function='cosine', train_set=0):
    """
    this code generate a list of mean error, the index of the element in the list is k-1
    """
    ame = [0 for i in range(len(k_range))]
    for user in user_list:
        for k in k_range:
             true, pred, mean_error = get_ame(user, k, test_df, train_df, 
                                              function=function, 
                                              train_set=train_set)
             ame[k-1] += mean_error
    return ame
         

def plot_best_k(ame):
    k_range = list(range(1, len(ame)+1))
    plt.plot(k_range, ame)

    plt.title('AME vs. K')
    plt.xlabel('K')
    plt.ylabel('Absolute Mean Error')
    plt.show()
    
    

def load_md(user, train=0, function='correlation', k=6):
    model_path = './models/train_{}/{}/{}/{}_k_{}.csv' \
            .format(train, function, str(user).zfill(7), str(user).zfill(7), k)
    md = pd.read_csv(model_path, sep=',', names=['song_id', 'score'], 
                     index_col=['song_id'])
    md = md.dropna()
    return md


def get_recall(user, test_df, md, N):
    """
    user: user id --> integer
    test_df: dataframe for the whole test set e.g. test_0.txt
    md: dataframe for target user's model
    """
    user_df = test_df.loc[user]
    truth = list(user_df['song_id'])
    pred = list(md.index)[:N]
    
    return len(find_intersection(pred, truth)) / len(truth)
    

def get_recalls(user_list, N_range, k_range, test_df, function='correlation'):

    if type(k_range) == int:
        k_range = list(k_range)

    recall = np.zeros((len(k_range), len(N_range)))
    
    for k in k_range:
        for user in user_list:
            rec = []
            md = load_md(user, train=0, function=function, k=k)
            for N in N_range:
                rec.append(get_recall(user, test_df, md, N))
            recall[k-1] += np.array(rec)

        recall[k-1] /= len(user_list)

    return recall


def get_coverage(user_list, N_range, k, train_df, function='correlation', train=0):
    item_num = len(list(set(train_df['song_id'])))
    cov = []
    for N in N_range:
        items = []
        for user in user_list:
            md = load_md(user, train=train, function=function, k=k)
            items += list(md.index[:N])
        cov.append(len(set(items)) / item_num)
    return cov
    
    
def get_coverages(user_list, N_range, k_range, train_df, function='correlation', train=0):
    if type(k_range) == int:
        k_range = list(k_range)
    item_num = len(list(set(train_df['song_id'])))
    coverage = np.zeros((len(k_range), len(N_range)))
    for k in k_range:
        cov = []
        for N in N_range:
            items = []
            for user in user_list:
                md = load_md(user, train=train, function=function, k=k)
                items += list(md.index[:N])
            cov.append(len(set(items)) / item_num)
        coverage[k-1] = cov
    return coverage
    

def plot_recalls(recall, N_range, k_range):
    legend = []
    for k in k_range:
        legend.append('k={}'.format(k))
        rec = recall[k-1]
        plt.plot(N_range, rec)
    plt.legend(legend)
    plt.title('Recall vs. N')
    plt.xlabel('N')
    plt.ylabel('Recall')
    plt.show()


def plot_coverages(coverages, N_range, k_range):
    legend = []
    for k in k_range:
        legend.append('k={}'.format(k))
        cov = coverages[k-1]
        plt.plot(N_range, cov)
    plt.legend(legend)
    plt.title('Coverage vs. N')
    plt.xlabel('N')
    plt.ylabel('Coverage')
    plt.show()
    

def plot_recall_and_coverage(recall, coverage, N_range, k):
    
    rec = recall[k-1]
    cov = coverage[k-1]
    
    plt.plot(N_range, rec)
    plt.plot(N_range, cov)
    plt.legend(['recall', 'coverage'])
    plt.title('Recall/Coverage vs. N')
    plt.xlabel('N')
    plt.ylabel('recall/coverage')
    plt.show()
    
#def diversity():
#    
#    for user in user_list:
#        
#        md = load_md(user, train=train, function=function, k=k)
        
        
    
    
k_range = list(range(1,21))
k = 6
user_list = list(range(280))
input_path = './dataset/train_0.txt'
train_df = load_dataset(input_path)
test_df = load_test(test_path='./dataset/test_0.txt')
item_list = list(set(train_df['song_id']))
item_num = len(item_list)
N_range = list(range(100,10000,100))

