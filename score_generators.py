# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:29:34 2019

@author: septe

This code contains all functions for score generator
you can also run this code to save scores for future use
to run this code, you need to provide k range, user list, and similarity functions

"""
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, kendalltau
from scipy.spatial.distance import euclidean, cosine, correlation
from sklearn.metrics import jaccard_similarity_score
from load_data import *


def sort_dict_by_value(d, reverse=False):
    """
    False: from lowest value to highest value
    True: from highest value to lowest value
    """
    return sorted(d.items(), key=lambda x: x[1], reverse=reverse)


def get_similarity(feature_vector, function='correlation'):
    """
    1. calculate similarity based on input function 
    2. rank neighbors according to similarity from high to low
    """

    sim_dict = {}
    rank = []
    for nb, vector in feature_vector.items():
        u_vector = vector[0]
        nb_vector = vector[1]

        if function == 'pearsonr': # same as correlation
            sim_dict[nb] = pearsonr(u_vector, nb_vector)[0]
                        
        elif function == 'kendalltau':
            sim_dict[nb] = kendalltau(u_vector, nb_vector)[0]
            
        elif function == 'correlation':
            sim_dict[nb] = 1 - correlation(u_vector, nb_vector, centered=False)
            
        elif function == 'cosine': # same as correlation when standardized
            sim_dict[nb] = 1 - cosine(u_vector, nb_vector)

        elif function == 'euclidean': # it is actually reverse euclidean
            sim_dict[nb] = 1/euclidean(u_vector, nb_vector)
        
    sim_tup = sort_dict_by_value(sim_dict, reverse=True)
    for tup in sim_tup:
        rank.append(tup[0])
    return sim_dict, rank


def KNN(rank, k=5):
    nearest_nb = rank[:k]
    return nearest_nb


def get_user_CF_weights(nearest_nb, initial_nb, sim_dict, user_song_list, 
                        rating_filter=False, sort=False):
    '''
    calculate weights for each song candidates based on user CF
    
    filter out songs which are never given a rate equal or above rating filter 
    value by any initial neighbors

    :param nearest_nb:
    :param df: dataframe of dataset
    :return:
    song_candidates = {song_id:weight, song_id:weight, ...}
    song_rank_UCF = [[song_id, weight], [song_id, weight], ...]
    '''
    song_candidates = {}

    for nb in nearest_nb:
        song_list = initial_nb[nb][0]
        rating_list = initial_nb[nb][1]
        
        # filter out low rating songs
        if rating_filter:
            i = 0
            while i < len(song_list):
                if rating_list[i] < rating_filter:
                    rating_list.pop(i)
                    song_list.pop(i)
                else:
                    i += 1
        
        for j in range(len(song_list)):
            if song_list[j] in user_song_list:
                continue
            elif song_list[j] in song_candidates.keys():
                song_candidates[song_list[j]] += rating_list[j] * sim_dict[nb]
            else:
                song_candidates[song_list[j]] = rating_list[j] * sim_dict[nb]

    song_candidate_list = list(song_candidates.keys())

    if sort:
        song_rank_UCF = sort_dict_by_value(song_candidates, reverse=True)
        return song_rank_UCF, song_candidate_list
    else:
        return song_candidates, song_candidate_list


def get_content_weights(song_candidate_list, user_song_list):
    """
    song_candidate_list is the union of songs for all nearest neighbors
    [song_id, song_id, ...]
    
    user_song_list is the song list of user's listening history
    [song_id, song_id, ...]
    
    ct is the dataframe loaded using load_cotent_info from load_data.py
    the header is ['song_id', 'album_id', 'artist_id', 'genre_id']
    the index is 'song_id'
    
    content weights for certain song candidate is sum of similarity between 
    song candidate and all songs in user song list
    
    song_rank_CT is a dictionary of song candidates to their weights
    {song_id:weight, song_id:weight, ...}
    
    similarity function: Jaccard
    
    """

    song_rank_CT = {}
            
    sim = np.zeros((len(song_candidate_list), len(user_song_list))) + 677
    song_list = list(set(song_candidate_list + user_song_list))
    
    while np.any(sim==677):
        song = np.array(song_list).min()
        a = np.load('./raw_feature/item_sim/item_jaccard_sim_{}.npy'.format(song))
        if song in song_candidate_list:
            for i in range(len(user_song_list)):
                sim[song_candidate_list.index(song)][i] = a[0][user_song_list[i] - song]
        
        if song in user_song_list:
            for j in range(len(song_candidate_list)):
                sim[j][user_song_list.index(song)] = a[0][song_candidate_list[j] - song]
                
        song_list.remove(song)
        
    for k in range(len(song_candidate_list)):
        song_rank_CT[song_candidate_list[k]] = sim[k].sum()
        
    return song_rank_CT


def get_popularity_weights(song_candidate_list, train=0):
    """
    assign weights to song candidates based on popularity in given dataset
    :param df: loaded dataframe of dataset
    :param song_candidate_list: 
    :return:
    """

    song_rank_pop = {}

    df = pd.read_csv('./raw_feature/popularity/popularity_{}.csv'.format(train))
    df.columns = ['song_id', 'popularity']
    df = df.set_index('song_id')

    for song in song_candidate_list:
        song_rank_pop[song] = df.loc[song]['popularity']

    return song_rank_pop


###############################################################################
user_list = list(range(100))
functions = ['cosine', 'kendalltau']
k_range = list(range(1, 101))
train_set = 0


# extract user list
# extract mean, std for all users
# they will be used while target user std == 0
df = pd.read_csv('./dataset/train_0.txt', sep='\t', header=None)
df.columns = ['user_id', 'song_id', 'rating']
#user_list = list(set(df['user_id']))
avg = df['rating'].mean()
std = df['rating'].std()
del df


def main():
    # load mean, std values for all user for feature standardization
    print('loading user statistics.......')
    stat = pd.read_csv('./raw_feature/new_train_{}_user_info.csv'
                       .format(train_set, train_set), sep=',')
    stat.columns = ['user_id', 'mean', 'std']
    stat = stat.set_index('user_id')
    
    for user in user_list:
        
        print('user {} || loading feature.......'.format(user))
        feature_vector, initial_nb, user_song_list = load_feature(user, train=train_set)
        print('user {} || standardizing feature.......'.format(user))
        stand_feature_vector = stand_feature(stat, user, feature_vector, avg=avg, std=std)
        
        for function in functions:
            
            print('user {}, {} similarity || calculating similarities.......'.format(user, function))
            sim_dict, rank = get_similarity(stand_feature_vector, function=function)
    
            for k in k_range:
    
                output = './scores/user-user/train_{}/{}/{}/{}/'\
                    .format(train_set, str(user).zfill(7), function, k)
                output1 = output + 'user_CF_{}.csv'.format(str(user).zfill(7))
                output2 = output + 'content_{}.csv'.format(str(user).zfill(7))
                output3 = output + 'popularity_{}.csv'.format(str(user).zfill(7))
    
                if os.path.exists(output3):
                    print('user {}, {} similarity, k = {} || scores have been saved.......'\
                          .format(user, function, k))
                    continue
                elif not os.path.exists(output):
                    os.makedirs(output)
                    
                print('user {}, {} similarity, k = {} || finding neighbors.......'\
                          .format(user, function, k))
                nearest_nb = KNN(rank, k=k)
                # print(nearest_nb)
    
                print('user {}, {} similarity, k = {} || calculating UCF scores.......'\
                          .format(user, function, k))
                song_rank_UCF, song_candidate_list = \
                    get_user_CF_weights(nearest_nb, initial_nb, sim_dict, user_song_list)
                # print(song_rank_UCF)
                
                print('user {}, {} similarity, k = {} || saving UCF scores.......'\
                          .format(user, function, k))
                ucf_df = pd.DataFrame.from_dict(song_rank_UCF, orient='index', columns=['UCF_score'])
                ucf_df.to_csv(output1)
            
                print('user {}, {} similarity, k = {} || calculating content scores.......'\
                          .format(user, function, k))
                song_rank_CT = get_content_weights(song_candidate_list, user_song_list, sim)
                
                print('user {}, {} similarity, k = {} || saving content scores.......'\
                          .format(user, function, k))
                ct_df = pd.DataFrame.from_dict(song_rank_CT, orient='index', columns=['CT_score'])
                ct_df.to_csv(output2)
                
                print('user {}, {} similarity, k = {} || calculating popularity scores.......'\
                          .format(user, function, k))
                song_rank_pop = get_popularity_weights(song_candidate_list, train=train_set)
                
                print('user {}, {} similarity, k = {} || saving popularity scores.......'\
                          .format(user, function, k))
                ucf_df = pd.DataFrame.from_dict(song_rank_pop, orient='index', columns=['popularity_score'])
                ucf_df.to_csv(output3)
                

#if __name__ == '__main__':
#    main()