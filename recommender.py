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



def load_scores(sim, k, u_id):
    score_path = './scores/{}_k{}_{}.npy'.format(sim, k, u_id)
    return np.load(score_path)


def recommender(sim, k, u_id):
    
    score = load_scores(sim, k, u_id)
    i_list = score[0]
    i_score = score[1]

