#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:10:02 2019

@author: zhechenw
"""

#from load_data import *
import os
import pandas as pd
import numpy as np

train_path = './dataset/train_0.txt'
item_content_path = './dataset/song-attributes.txt'
item_genre_hierarchy_path = './dataset/genre-hierarchy.txt'

train_df = pd.read_csv(train_path, 
                       sep='\t', 
                       names=['u_id', 'i_id', 'rating'])

item_content_df = pd.read_csv(item_content_path, 
                              sep='\t', 
                              names=['i_id', 'album_id', 'artist_id', 'genre_id'])

u_dist = train_df.groupby('u_id').size()
i_dist = train_df.groupby('i_id').size()


def get_record(df, t_id, name='i_id'):
    """
    df: dataframe with the index set to target id
    t_id: target id, it is user id (u_id) while doing user based CF
    name: column name for the record, it is 'i_id' while doing user based CF
    return a 1-D numpy array of record
    """
    return df.loc[t_id][name].to_numpy()


def find_intersection(array1, array2):
    """
    return intersection in form of 1-D array
    """
    return np.intersect1d(array1, array2, assume_unique=True)


def find_initial_neighbor(train_df, limit=100):
    
    u_list = train_df.index.unique().to_numpy()
    u_num = len(u_list)
    u_map = {}
    inb_map = {}
    
    for user in u_list:
        u_map[user] = get_record(train_df, user)

    for i in range(u_num):
        for j in range(i+1, u_num):
            record1 = u_map[u_list[i]]
            record2 = u_map[u_list[j]]
            intersection = np.intersect1d(record1, record2, assume_unique=True)
            inb_map[u_list[i]] = intersection
            
            
            
            
            
    