#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:16:59 2019

@author: zhechenw
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_processing import *
from trainer import *
from tqdm import tqdm


test_path = './dataset/test_0.txt'
train_path = './dataset/train_0.txt'
contents_path = './dataset/song-attributes.txt'

train_df = load_dataset(train_path)
item_content_df = load_item_contents(contents_path)
test_df = load_dataset(test_path)
test_df = test_df.set_index('u_id')

U = Users(train_df)
u_list = U.u_list
I = Items(item_content_df, train_df)


def get_result(u_id, k, sim_fn, weights):
    recommend_dict = recommender(u_id, k, sim_fn, weights)
    test_dict = test_df.loc[u_id].to_dict()['rating']
    
    return recommend_dict, test_dict
    

def get_recall(u_id, k, sim_fn, weights, n=None):
    r, t = get_result(u_id, k, sim_fn, weights)
    r = dict(list(r.items())[:n])
    inte = np.intersect1d(r.keys(), t.keys(), assume_unique=True)
    
    return inte.size / len(t.keys())


def get_precision(u_id, k, sim_fn, weights, n=None):
    r, t = get_result(u_id, k, sim_fn, weights)
    r = dict(list(r.items())[:n])
    inte = np.intersect1d(r.keys(), t.keys(), assume_unique=True)
    
    return inte.size / len(r.keys())


def get_recalls(u_ids, k, sim_fn, weights, n=None):
    
    recommended_i = []
    test_i = []
    
    for u_id in u_ids:
        r, t = get_result(u_id, k, sim_fn, weights)
        recommended_i += list(r.keys())[:n]
        test_i += list(t.keys())
        
    return len(set(recommended_i).intersection(test_i)) / len(set(test_i))
    
    
def get_precisions(u_ids, k, sim_fn, weights, n=None):
    
    recommended_i = []
    test_i = []
    
    for u_id in u_ids:
        r, t = get_result(u_id, k, sim_fn, weights)
        recommended_i += list(r.keys())[:n]
        test_i += list(t.keys())
        
    return len(set(recommended_i).intersection(test_i)) / len(set(recommended_i))
    

def get_diversity(u_ids, k, sim_fn, weights, I, n=None):
    
    div = 0
    
    for u_id in tqdm(u_ids):
        r = list(recommender(u_id, k, sim_fn, weights).keys())[:n]
        sim = 0
        
        for i in range(len(r)):
            f1 = I.get_item(r[i]).contents
            
            for j in range(i+1, len(r)):
                try:
                    sim += load_item_sim(r[i], r[j])
                except:
                    f2 = I.get_item(r[j]).contents
                    sim += jaccard_score(f1, f2, average='micro')
        
        div += 1 - sim / ((len(r) - 1) * len(r) / 2)
        
    return div / len(u_ids)


def get_gini(u_ids, k, sim_fn, weights, n=None, plot=True):
    
    r = []
    
    for u_id in u_ids:
        
        recommend_dict = recommender(u_id, k, sim_fn, weights)
        r += list(recommend_dict.keys())[:n]
    
    p_map = {}
    
    for i in r:
        p_map[i] = r.count(i)
        
    p_map = dict(sorted(p_map.items(), key=lambda x: x[1]))
    p = list(p_map.keys())
    gini = 0
    n = len(r)
    
    if plot:
        plt.plot(range(len(p)), list(p_map.values()))
        plt.title('Popularity Distribution', fontsize=12)
        plt.xlabel('Popularity rank, low to high', fontsize=12)
        plt.ylabel('Popularity', fontsize=12)
        plt.show()
    
    else:
        
        for i in r:
            gini += (2 * p.index(i) - n - 1) * p_map[i]
            
        return gini / (n - 1)
        

def get_coverage(u_ids, k, sim_fn, weights, I, n=None):
    r = []
        
    for u_id in u_ids:
        
        recommend_dict = recommender(u_id, k, sim_fn, weights)
        r += list(recommend_dict.keys())[:n]
        
    return len(set(r)) / len(I.i_list)



