#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 21:06:20 2019

@author: zhechenw
"""
import os
import numpy as np
from tqdm import tqdm
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
I = Items(item_content_df, train_df)

similarity_functions = [kendalltau, cosine]
k_range = range(1, 10)
u_list = U.u_list[:5]


for u_id in u_list:
    for k in k_range:
        for sim_fn in similarity_functions:
            if sim_fn == kendalltau:
                sim = 'ken'
            else:
                sim = 'cos'
                
            output = './scores/{}_k{}_{}'.format(sim, k, u_id)           

            if os.path.exists(output + '.npy'):
                print('\nuser {}, k={}, sim={} scores already saved!'.format(u_id, k, sim))
                continue
            
            print('UCF score calculating.......')
            ucf_score = get_UCF_score(u_id, U, k=k, limit=100, sim_fn=sim_fn)
            
            candidates = get_candidates(ucf_score)
            
            print('CF score calculating.......')
            cf_score = get_CF_score(u_id, candidates, U, I)
            
            print('POP score calculating.......')
            pop_score = get_POP_score(u_id, candidates, I)
            
            score = get_final_score(ucf_score, cf_score, pop_score, weights=None)
            
            score_map = np.array((candidates, score.T[0], score.T[1], score.T[2]))    
            
            np.save(output, score_map)
        
            print('\nuser {}, k={}, sim={} scores saved!'.format(u_id, k, sim))
    
    

        


