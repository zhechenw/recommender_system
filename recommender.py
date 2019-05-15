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


def getKey(item):
    return item[0]


def recommend(song_rank_UCF, song_rank_CT, song_rank_pop, 
              weights):
    '''
    rank song candidates based on input weights and recommend accordingly
    :param n: number of songs to recommend
    :param user_CF_weights:
    :param popularity_weights:
    :param content_filtering_weights:
    :return:
    '''
    # sort weight by song id
    # song_rank_UCF = sorted(song_rank_UCF, key=getKey) # tuple
    song_rank_UCF = dict(sorted(song_rank_UCF.items()))  # dict
    song_rank_CT = dict(sorted(song_rank_CT.items())) # dict
    song_rank_pop = dict(sorted(song_rank_pop.items())) # dict
    
    # extract weight
    # weight_UCF = []
    # for item in song_rank_UCF:
    #     weight_UCF.append(item[1])
    weight_UCF = list(song_rank_UCF.values())
    weight_CT = list(song_rank_CT.values())
    weight_pop = list(song_rank_pop.values())
    
    # extract song id
    song_list = list(song_rank_CT.keys())
    
    # create weight matrix
    weight_matrix = np.array([weight_UCF, weight_CT, weight_pop]).T
    scaler = StandardScaler()
    scaler.fit(weight_matrix)
    normalized_weight_matrix = scaler.transform(weight_matrix)

    weight_df = pd.DataFrame(normalized_weight_matrix)
    weight_df.columns = ['user_based_CF_weights',
                         'content_filtering_weights',
                         'popularity_weight']
    weight_df.index = song_list
    score = weight_df.dot(np.array(weights).reshape(3,1))
    score = score.iloc[:,0]
    recommend_df = score.sort_values(ascending=False)
    recommend_df = recommend_df.dropna()

    return recommend_df


train_set = 0
user_list = list(range(100)) # change here
functions = ['cosine', 'kendalltau']
k_list = list(range(1,101))
weights = [0.8, 0.1, 0.1]


def main():

    for user in user_list:
        
        for function in functions:
            
            for k in k_list:
                
                # create output path
                output_u = './models/train_{}/{}/{}/'.format(train_set, str(user).zfill(7), function)
                output_path = output_u + '{}_k_{}.csv'.format(str(user).zfill(7), k)
                
                # check if the model has already been saved
                if os.path.exists(output_path):
                    print('user {}, {} similarity, k = {} || model has already been saved!!!!!!!'
                          .format(user, function, k))
                    continue
                elif not os.path.exists(output_u):
                    os.makedirs(output_u)
                
                # load scores
                print('user {} || loading scores.......'.format(user))
                song_rank_UCF, song_rank_CT, song_rank_pop = load_score(user, 
                                                                        target_type='user', 
                                                                        train=train_set, 
                                                                        function=function,
                                                                        k=k)
                print('user {} || scores loaded!!!!!!!'.format(user))
        
                # recommend 
                print('user{}, {} similarity, k = {} || generating model.......'
                          .format(user, function, k))
                recommend_df = recommend(song_rank_UCF, song_rank_CT, song_rank_pop,
                                         weights=weights)
        
                recommend_df.to_csv(output_path)
        
                print('user{}, {} similarity, k = {} || model saved!!!!!!!'
                          .format(user, function, k))
            

if __name__ == '__main__':
    main()