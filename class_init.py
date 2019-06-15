#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 20:04:27 2019

@author: zhechenw
"""
from sklearn.metrics import jaccard_similarity_score


class user:
    def __init__(self, u_id, train_df_u):
        """
        u_id is the user id
        train_df_u is the training dataframe set index to u_id
        """
        self.u_id = u_id
        self.record = train_df_u.loc[u_id].to_numpy().T
        self.i_list = self.record[0]
        self.i_num = len(self.i_list)
        self.rating = self.record[1]
        self.rating_avg = self.rating.mean()
        self.rating_std = self.rating.std()
        
        
    def diversity(self, item_content_df, sim_fn='Jaccard'):
        
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
        
    
    def gini_index(self):
        pass
    
    
    def i_nb(self):
        return initial_nb(self.u_id)
    
    
class initial_nb:
    def __ini
        
        
    
    
    
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

    
    
    
    
    
    
    
    
    
    
    
    
    