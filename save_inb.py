#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:36:37 2019

@author: zhechenw
"""
import json, os
from tqdm import tqdm
import numpy as np
from data_processing import *
from trainer import *
from scipy.stats import pearsonr, kendalltau
from scipy.spatial.distance import euclidean, cosine, correlation
from sklearn.preprocessing import StandardScaler

train_path = './dataset/train_0.txt'
contents_path = './dataset/song-attributes.txt'

train_df = load_dataset(train_path)
item_content_df = load_item_contents(contents_path)

U = Users(train_df)
u_list = U.u_list

for i in tqdm(range(len(u_list))):
    output = './initial_nb/inb{}.json'.format(u_list[i])
    if os.path.exists(output):
        print('\n user {} initial neighbor already saved!'.format(u_list[i]))
        continue
    
    u1 = U.get_user(u_list[i])
    inb_temp = {}
    if u1.rating_std == 0:
        u1.rating_std = U.rating_std
        u1.rating_avg = U.rating_avg
    
    for j in range(i+1, len(u_list)):
        u2 = U.get_user(u_list[j])
        if u2.rating_std == 0:
            u2.rating_std = U.rating_std
            u2.rating_avg = U.rating_avg
            
        intersect = np.intersect1d(u1.i_list, u2.i_list, assume_unique=True)
        f1, f2 = [], []
        
        if len(intersect) > 2: # kendaltau raise error when tersect <= 2
            for song in intersect:
                f1.append((u1.record[song] - u1.rating_avg) / u1.rating_std)
                f2.append((u2.record[song] - u2.rating_avg) / u2.rating_std)
            
            inb_temp[int(u2.u_id)] = (f1, f2)
    
    sorted_inb = sorted(inb_temp.items(), key=lambda x: len(x[1][0]), reverse=True)
#    inb_map[u1.u_id] = sorted_inb
    with open(output, 'w') as f:
        json.dump(sorted_inb, f)
    print('user {} initial neighbor saved!'.format(u_list[i]))    
    