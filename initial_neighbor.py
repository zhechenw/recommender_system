# -*- coding: utf-8 -*-
"""
Created on Mon May 13 13:18:41 2019

@author: septe
"""
from load_data import *
import os


train = 'train_0'
target = 'user'
input_path = './dataset/{}.csv'.format(train)
initial_neighbor_num = 100


def get_initial_neighbor(df, target_id, target, target_list, overlap_limit=None,
                         dynamic_limit=100, progress=False):
    """
    find initial neighbors with overlap limit
    
    target indicates it is user-user CF or item-item CF
    
    :param df: 
        input dataframe
        usually get from load_dataset, 
        if not make sure headers are defined as 'user_id', 'song_id', 'rating'
        
    :param target_user: 
        target user id --> int
        
    :param overlap_limit: 
        It is the number of overlap in listening history between target user
        and initial neighbor candidate. Only when the overlap is greater than
        this overlap_limit the candidate are defined as initial neighbor.
        
        It is default as None, in other word, the function is default to use 
        dynamic limit.
        
        if you want to define a specific overlap limit, dynamic_limit has to 
        set as None
        
        if given as int
            the initial neighbor will be users with overlap greater than this 
            limit
        if given as fraction (float)
            the actual overlap limit will be the given fraction of the target
            user's listening history
            example: user id 0's listening history has 100 songs, given 
            fraction 0.5, the actual overlap limit is 50
            
    :param dynamic_limit:
        it is the number of initial neighbor
        it is default to be 100
        this make sure every target user have enough number of initial neighbor 
        for future use
        if the target user's initial neighbor candidates is less than 100, it 
        will use all user's initial neighbor candidates as initial neighbors
        
    :param standardize:
        let it be False, don't change!!!
        
    :return:
    initial_nb = {nb_id:[[song_id, ...], [rating, ...]], ...}
    feature_vector = {nb_id:[[r,r,r,...], [r,r,r,...]], nb_id:[[],[]], ...}
    """
    
    if target == 'user':
        obj = 'item'
    elif target == 'item':
        obj = 'user'
    else:
        raise ValueError('target has to be either user or item!!!!!!!')
        
    df.columns=['user','item','rating']
    df = df.set_index([target, obj])
    target_info = df.loc[target_id]
    target_obj_list = list(target_info.index)
    
    initial_nb = {}
    feature_vector = {}

    if dynamic_limit:
        if progress:
            print('{} {} || applying dynamic overlap limit.......'.format(target, target_id))
        raw_nb_dict = {}
        initial_nb_list = []

        for nb in target_list:

            if progress:
                print ('{} {} || checking initial neighbor candidates {}/{}.......'
                       .format(target, target_id, target_list.index(nb)+1, len(target_list)))
            if nb != target_id:
                # neighbor candidate dataframe
                nb_info = df.loc[nb]
                # item list when 'user', user list when 'item'
                nb_obj_list = list(nb_info.index)
                overlap = find_intersection(target_obj_list, nb_obj_list)
                overlap_num = len(overlap)

                if overlap_num > 0:
                    # raw candidate
                    raw_nb_dict[nb] = overlap_num

        # sort by overlap number
        sorted_raw_nb_tuple = sorted(raw_nb_dict.items(), key=lambda x: x[1], 
                                     reverse=True)
        
        # when initial neighbor candidates number < dynamic limit
        if len(sorted_raw_nb_tuple) < dynamic_limit:
            dynamic_limit = len(sorted_raw_nb_tuple)
            
        for i in range(dynamic_limit):
            initial_nb_list.append(sorted_raw_nb_tuple[i][0])

        for nb in initial_nb_list:
            
            if progress:
                print('{} {} || finalizing initial neighbor {}/{}.......'
                      .format(target, target_id, initial_nb_list.index(nb)+1, dynamic_limit))
                
            nb_info = df.loc[nb]
            nb_obj_list = list(nb_info.index)
            overlap = find_intersection(target_obj_list, nb_obj_list)
            # create initial neighbor dict
            nb_rating_list = list(nb_info['rating'])
            initial_nb[nb] = [nb_obj_list, nb_rating_list]
            # extract ratings as feature vector
            target_rating = []
            nb_rating = []

            for item in overlap:
                target_rating.append(int(target_info.loc[item]))
                nb_rating.append(int(nb_info.loc[item]))

            feature_vector[nb] = [target_rating, nb_rating]
            
        raw = len(raw_nb_dict)
        filtered = len(initial_nb)
        
        return feature_vector, initial_nb, raw, filtered, target_obj_list

    else:
        raw = 0
        if progress:
            print('applying static overlap limit.......')
        for nb in target_list:
            if nb != target_id:
                nb_info = df.loc[nb]
                nb_obj_list = list(nb_info.index)
                overlap = find_intersection(target_obj_list, nb_obj_list)
                overlap_num = len(overlap)

                if type(overlap_limit) == float:
                    overlap_limit = round(len(target_obj_list) * overlap_limit)

                if overlap_num > 0:
                    raw += 1

                    if overlap_num > overlap_limit-1:
                        nb_rating_list = list(nb_info['rating'])
                        initial_nb[nb] = [nb_obj_list, nb_rating_list]
                        # extract ratings as feature vector
                        target_rating = []
                        nb_rating = []

                        for item in overlap:
                            target_rating.append(int(target_info.loc[item]))
                            nb_rating.append(int(nb_info.loc[item]))

                        feature_vector[nb] = [target_rating, nb_rating]

        filtered = len(initial_nb)

        return feature_vector, initial_nb, raw, filtered, target_obj_list


def main():

    df = load_dataset(input_path)
    target_list = get_target_list(df, target)
    print('dataset loading complete!!!!!!!')

    for user in target_list:

        output = './raw_feature/user-user/{}/dynamic_overlap_limit/{}/'.format(train, str(user).zfill(7))
        output1 = output + 'feature_{}.txt'.format(str(user).zfill(7))
        output4 = output + 'initial_nb_{}.txt'.format(str(user).zfill(7))
        output5 = output + 'user_song_list_{}.txt'.format(str(user).zfill(7))

        if os.path.exists(output1):
            print('user {} raw feature is already saved.......'.format(user))
            continue

        if not os.path.exists(output):
            os.makedirs(output)

        print('=================processing user {}==========================='.format(user))

        # save feature
        feature_vector, initial_nb, raw, filtered, user_song_list = \
            get_initial_neighbor(df, user, target, target_list, 
                                 dynamic_limit=initial_neighbor_num, 
                                 progress=True)
        print('user {} feature extraction complete!!!!!!!'.format(user))

        f1 = open(output1, '+w')
        for nb in feature_vector.keys():
            f1.write(str(nb) + '\n')
            r1 = feature_vector[nb][0]
            r2 = feature_vector[nb][1]

            for r in r1:
                f1.write(str(r) + '\t')

            f1.write('\n')

            for r in r2:
                f1.write(str(r) + '\t')

            f1.write('\n')

        f1.close()
        print('user {} feature vector saved!!!!!!!'.format(user))

        f2 = open(output4, '+w')
        for nb in initial_nb.keys():
            f2.write(str(nb) + '\n')
            sl = initial_nb[nb][0] # song list
            rr = initial_nb[nb][1] # raw rating

            for s in sl:
                f2.write(str(s) + '\t')

            f2.write('\n')

            for r in rr:
                f2.write(str(r) + '\t')

            f2.write('\n')

        f2.close()
        print('user {} raw initial neighbor info saved!!!!!!!'.format(user))

        f3 = open(output5, '+w')
        for song in user_song_list:
            f3.write(str(song) + '\n')
        f3.close()
        print('user {} song list saved!!!!!!!'.format(user))


if __name__ == '__main__':
    main()