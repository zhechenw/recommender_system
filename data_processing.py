#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 20:04:27 2019

@author: zhechenw
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import jaccard_score


def load_dataset(path):
    return pd.read_csv(path, sep='\t', names=['u_id', 'i_id', 'rating'])


def load_item_contents(path):
    df = pd.read_csv(path, sep='\t', names=['i_id', 'album_id', 'artist_id', 'genre_id'])
    return df.set_index('i_id')


class Users:
    def __init__(self, train_df):

        self.u_list = train_df['u_id'].unique()
        self.rating_avg = train_df['rating'].mean()
        self.rating_std = train_df['rating'].std()
        
        train_df = train_df.set_index(['u_id', 'i_id'])
        
        try:
            with open('./user_records.json', 'r') as f:
                self.records = {int(k):v for k, v in json.load(f).items()}
            
        except:
            self.records = {}
            for u in tqdm(self.u_list):
                record = train_df.loc[u].to_dict()['rating']
                self.records[int(u)] = record
            
            with open('./user_records.json', 'w') as f:
                json.dump(self.records, f)
    
    
    def get_user(self, u_id):
        return User(u_id, self.records)   
    
  
class User:
    def __init__(self, u_id, records):
        self.u_id = u_id
        self.record = {int(k):v for k, v in records[self.u_id].items()} 
        self.i_list = list(self.record.keys())
        self.i_num = len(self.i_list)
        self.rating = np.array(list(self.record.values()))
        self.rating_avg = self.rating.mean()
        self.rating_std = self.rating.std()
        
        
    def get_dist(self):
        dist = []
        ratings = list(self.record.values())
        
        for r in [1,2,3,4,5]:
            
            dist.append(ratings.count(r))
        
        return dist
    
    
    def get_diversity(self, item_content_df):
        
        sim = 0

        for i in range(self.i_num):
            i_id1 = self.i_list[i]
            item1 = item(i_id1, item_content_df)
            for j in range(i+1, self.i_num):
                i_id2 = self.i_list[j]
                item2 = item(i_id2, item_content_df)
                sim += jaccard_score(item1.contents, item2.contents, average='micro')
                    
        return 1 - sim / (((self.i_num - 1) * self.i_num) / 2)
    

    def get_gini_index(self):
        pass
   
    
class Items:
    def __init__(self, item_content_df, train_df):
        self.contents_matrix = item_content_df.to_numpy()
        self.df = train_df.set_index('i_id')
        self.i_list = list(self.df.index)
        self.rating_avg = self.df['rating'].mean()
        self.rating_std = self.df['rating'].std()
        self.pop_map = train_df.groupby('i_id').size().to_dict()
        
        
    def get_item(self, i_id):
        return item(i_id, self.contents_matrix, self.pop_map)
        
        
    def get_rating_avg(self, i_id):
        return self.df.loc[i_id]['rating'].mean()
    
    
    def get_rating_std(self, i_id):
        return self.df.loc[i_id]['rating'].std()
        
    
    def diversity(self, sim_fn=jaccard_score):
        
        i_num = len(self.contents_matrix)
        sim = 0
            
        for i in range(i_num):
            f1 = self.contents_matrix[i]
            for j in range(i+1, i_num):
                f2 = self.contents_matrix[j]
                sim += sim_fn(f1, f2, average='micro')
                    
        return sim / (((1 + i_num) * i_num) / 2)
    
    
    def get_gini(self, plot=True):
        p_map = dict(sorted(self.pop_map.items(), key=lambda x: x[1]))
        p = list(p_map.keys())

        if plot:
            plt.plot(range(len(p)), list(p_map.values()))
            plt.title('Popularity Distribution', fontsize=12)
            plt.xlabel('Popularity rank, low to high', fontsize=12)
            plt.ylabel('Popularity', fontsize=12)
            plt.show()
        
        else:
            i_list = I.i_list
            gini = 0
            n = len(i_list)
            for i in i_list:
                gini += (2 * p.index(i) - n - 1) * p_map[i]
            
            return gini / (n - 1)
    

class item:
    def __init__(self, i_id, contents_matrix, pop_map):
        """
        i_id is the item id
        train_df_i is the training dataframe set index to i_id
        item_content_df is the song-attribute dataframe with index of i_id
        """
        self.i_id = i_id          
        self.contents = contents_matrix[i_id]
        self.album_id = self.contents[0]
        self.artist_id = self.contents[1]
        self.genre_id = self.contents[2]
        self.popularity = pop_map[self.i_id]    
        
        

        
        
        
        
        






    
    
    
    
    
    
    
    
    
    
    
    
    