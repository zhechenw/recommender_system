#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 20:04:27 2019

@author: zhechenw
"""
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, kendalltau
from scipy.spatial.distance import euclidean, cosine, correlation
from sklearn.metrics import jaccard_similarity_score


class Users:
    def __init__(self, train_df):

        self.u_list = train_df['u_id'].unique()
        self.rating_avg = train_df['rating'].mean()
        self.rating_std = train_df['rating'].std()
        
        train_df = train_df.set_index(['u_id', 'i_id'])
        self.records = {}
        
        for u in self.u_list:
            record = train_df.loc[u].to_dict()['rating']
            self.records[u] = record
    
    
    def get_user(self, u_id):
        return User(u_id, self.records)

    
    def i_nb(self):
        return initial_nb(self.u_id)
    

class User:
    def __init__(self, u_id, records):
        self.u_id = u_id
        self.record = records[self.u_id]
        self.i_list = np.array(list(self.record.keys()))
        self.i_num = len(self.i_list)
        self.rating = np.array(list(self.record.values()))
        self.rating_avg = self.rating.mean()
        self.rating_std = self.rating.std()
    
    
    def get_initial_nb(self, limit=100):
        pass
        
        
    def get_diversity(self, item_content_df, sim_fn='Jaccard'):
        
        sim = 0
        
        if sim_fn == 'Jaccard':
            similarity = jaccard_similarity_score
            
        for i in range(self.i_num):
            i_id1 = self.i_list[i]
            item1 = item(i_id1, item_content_df)
            for j in range(i+1, self.i_num):
                i_id2 = self.i_list[j]
                item2 = item(i_id2, item_content_df)
                sim += similarity(item1.contents, item2.contents)
                    
        return sim / (((1 + self.i_num) * self.i_num) / 2)
    

    def get_gini_index(self):
        pass
    
        
class Initial_NB:
    def __init__(self, U):

        self.inb_map = {}
        
        for i in range(len(U.u_list)):
            u1 = U.get_user(U.u_list[i])
            inb_temp = {}
            
            for j in range(i+1, len(U.u_list)):
                u2 = U.get_user(U.u_list[j])
                intersect = np.intersect1d(u1.i_list, u2.i_list, assume_unique=True)
                f1, f2 = [], []
                
                if len(intersect) > 0:
                    for song in intersect:
                        with np.errstate(divide='raise'):
                            try:
                                f1.append((u1.record[song] - u1.rating_avg) / u1.rating_std)
                            except FloatingPointError:
                                f1.append((u1.record[song] - U.rating_avg) / U.rating_std)
                            try:
                                f2.append((u2.record[song] - u2.rating_avg) / u2.rating_std)
                            except FloatingPointError:
                                f2.append((u2.record[song] - U.rating_avg) / U.rating_std)
                    
                    inb_temp[u2.u_id] = (len(intersect), f1, f2)
            
            sorted_inb = sorted(inb_temp.items(), key=lambda x: x[1][0], reverse=True)
            self.inb_map[u1.u_id] = sorted_inb
        
        
    def get_inb(self, u_id, limit=100):
        return self.inb_map[u_id][:limit]
    
    
    def get_similarity(self, u_id, limit=100, sim_fn=kendalltau):
        self.sim = {}
        inb = self.inb_map[u_id][:limit]
        for i in range(len(inb)):
            inb_id = inb[i][0]
            f1 = inb[i][1][1]
            f2 = inb[i][2][2]
            self.sim[inb_id] = sim_fn(f1, f2)
        
        
        
class Items:
    def __init__(self):
        
    
    
class item:
    def __init__(self, i_id, item_content_df, stat=False, train_df_i=None):
        """
        i_id is the item id
        train_df_i is the training dataframe set index to i_id
        item_content_df is the song-attribute dataframe with index of i_id
        """
        self.i_id = i_id          
        self.contents = item_content_df.loc[self.i_id].to_numpy()
        self.album_id = self.contents[0]
        self.artist_id = self.contents[1]
        self.genre_id = self.contents[2]
            
        if stat and train_df_i:
            self.record = train_df_i.loc[self.i_id].to_numpy().T
            self.u_list = self.record[0]
            self.rating = self.record[1]
            self.rating_avg = self.rating.mean()
            self.rating_std = self.rating.std()
    
    
    def global_rating_avg(train_df):
        return train_df['rating'].mean()
        
    
    def global_rating_std(train_df):
        return train_df['rating'].std()
        
    
    def popularity(train_df):
        return train_df.groupby('i_id').size().to_dict()


    def diversity(item_content_df, sim_fn='Jaccard'):
        
        i_list = item_content_df.index.to_numpy()
        i_num = len(i_list)
        sim = 0
        
        if sim_fn == 'Jaccard':
            similarity = jaccard_similarity_score
            
        for i in range(i_num):
            i_id1 = i_list[i]
            item1 = item(i_id1, item_content_df)
            for j in range(i+1, i_num):
                i_id2 = i_list[j]
                item2 = item(i_id2, item_content_df)
                sim += similarity(item1.contents, item2.contents)
                    
        return sim / (((1 + i_num) * i_num) / 2)

    
    
    
    
    
    
    
    
    
    
    
    
    