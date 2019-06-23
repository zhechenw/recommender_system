# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:26:13 2019

@author: septe

this code contains function to generate 
"""
import os
import numpy as np
import pandas as pd
from score_generators import *
from sklearn.preprocessing import StandardScaler

u_id = 0
k = 5
sim_fn = kendalltau
top_n = 10
weights = [0.8, 0.1, 0.1]

output = './models/{}_k{}_{}




def load_scores(u_id, k, sim_fn):
    
    score_path = './scores/{}_k{}_{}.npy'.format(sim_fn, k, u_id)
    return np.load(score_path)


def recommender(u_id, k, sim_fn, weights, n):
    
    try:
        score = load_scores(u_id, k, sim_fn)
        i_list = score[0]
        ucf = score[1]
        cf = score[2]
        pop = score[3]
        scores = np.delete(score, 0, 0).T.dot(weights)
        score_map = {int(k):v for k, v in dict(np.array((i_list, scores)).T).items()}
        
        return sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:n]
        
    except:
        print('loading scores failed, check score file!')    
    
    
    

