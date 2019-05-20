# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:13:45 2019

@author: septe

"""

import pandas as pd
import numpy as np


def find_intersection(list1, list2):
    """
    return intersection of two input lists
    """
    return list(set(list1).intersection(set(list2)))


def find_union(list1, list2):
    """
    return union of two input lists
    """
    return list(set(list1) | set(list2))


def convert_list_dtype(l, dtype=float):
    return list(map(dtype, l))


def load_dataset(input_path):
    """
    load dataset at given input path
    return dataframe of given dataset
    it can be used to load either training set or test set
    headers are defined as 'user_id', 'song_id', 'rating'
    """
    header = ['user_id', 'song_id', 'rating']
    df = pd.read_csv(input_path, sep='\t', names=header)
    return df


def get_target_list(df, target):
    df.columns = ['user', 'item', 'rating']
    df = df.set_index(target)
    target_list = list(set(df.index))
    return target_list


def load_content_info(path='./dataset/song-attributes.txt'):
    """
    load content information and return its dataframe
    """
    header = ['song_id', 'album_id', 'artist_id', 'genre_id']
    ct = pd.read_csv(path, sep='\t', names=header)
    ct = ct.set_index('song_id')
    return ct


def load_feature(target, target_type='user', train=0):
    """
    load feature vectors for initial neighbors

    """
    output = './raw_feature/{}-{}/train_{}/dynamic_overlap_limit/{}/'\
        .format(target_type, target_type, train, str(target).zfill(7))

    f1 = open(output + 'feature_{}.txt'.format(str(target).zfill(7))).read()
    f2 = open(output + 'initial_nb_{}.txt'.format(str(target).zfill(7))).read()
    f3 = open(output + 'user_song_list_{}.txt'.format(str(target).zfill(7))).read()

    f1 = f1.strip()
    f2 = f2.strip()
    f3 = f3.strip()
    f1 = f1.split('\n')
    f2 = f2.split('\n')
    f3 = f3.split('\n')

    nb_num = len(f1) // 3
    initial_nb = {}
    feature_vector = {}
    user_song_list = []

    for i in range(nb_num):
        nb_id = int(f1[i * 3])
        us_feature = f1[i * 3 + 1].strip().split('\t')
        nb_feature = f1[i * 3 + 2].strip().split('\t')
        us_feature = convert_list_dtype(us_feature, dtype=float)
        nb_feature = convert_list_dtype(nb_feature, dtype=float)
        feature_vector[nb_id] = [us_feature, nb_feature]

        nb_song = f2[i * 3 + 1].strip().split('\t')
        nb_rating = f2[i * 3 + 2].strip().split('\t')
        nb_song = convert_list_dtype(nb_song, dtype=int)
        nb_rating = convert_list_dtype(nb_rating, dtype=int)
        initial_nb[nb_id] = [nb_song, nb_rating]

    for song in f3:
        user_song_list.append(int(song))

    return feature_vector, initial_nb, user_song_list


def stand_feature(stat, user, feature_vector, avg=None, std=None):
    mean1, std1 = list(stat.loc[user])
    if avg and std and std1 == 0:
        mean1, std1 = avg, std
    nb_list = list(feature_vector.keys())
    stand_feature_vector = {}
    for nb in nb_list:
        mean2, std2 = list(stat.loc[nb])
        if avg and std and std2 == 0:
            mean2, std2 = avg, std
        v1 = (np.array(feature_vector[nb][0]) - mean1) / std1
        v2 = (np.array(feature_vector[nb][1]) - mean2) / std2
        stand_feature_vector[nb] = [list(v1), list(v2)]
    return stand_feature_vector


def load_score(target, target_type='user', train=0, function='correlation', k=6):

    path = './scores/{}-{}/train_{}/{}/{}/{}/'.format(target_type, target_type, 
                     train, str(target).zfill(7), function, k)
        
    f1 = pd.read_csv(path + 'user_CF_{}.csv'.format(str(target).zfill(7)), sep=',')
    f2 = pd.read_csv(path + 'content_{}.csv'.format(str(target).zfill(7)), sep=',')
    f3 = pd.read_csv(path + 'popularity_{}.csv'.format(str(target).zfill(7)), sep=',')

    song_rank_UCF = f1.set_index('Unnamed: 0').to_dict()['UCF_score']
    song_rank_CT = f2.set_index('Unnamed: 0').to_dict()['CT_score']
    song_rank_pop = f3.set_index('Unnamed: 0').to_dict()['popularity_score']

    return song_rank_UCF, song_rank_CT, song_rank_pop