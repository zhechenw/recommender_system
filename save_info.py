import os, math
import numpy as np
import pandas as pd
from load_data import *
from sklearn.metrics import jaccard_similarity_score


def get_item_info(df, dataset='train_0'):
    
    df = df.set_index('song_id')
    item_list = list(set(df.index))
    
    output = './raw_feature/item-item/{}/'.format(dataset)

    if not os.path.exists(output):
        os.makedirs(output)

    info_dict = {}
    for item in item_list:
        print('extracting item {}/{} information.......'.format(item+1, len(item_list)))
        info = df.loc[item]
        avg = np.mean(list(info['rating']))
        std = np.std(list(info['rating']))
        info_dict[item] = [avg, std]

    info_df = pd.DataFrame.from_dict(info_dict, orient='index')
    info_df.columns = ['mean', 'std']
    print(info_df.head())

    info_df.to_csv(output + '{}_item_info.csv'.format(dataset))


def get_user_info(df, dataset='train_0', user=None):
    """
    this code generate mean and std for user rating
    it can be applied to single certain user and return its info dataframe, mean, and std
    or to all users in a given dataset (default train_0) and save all info as csv file
    """
    df = df.set_index('user_id')
    user_list = list(set(df.index))

    if user:
        print('extracting user {} information.......'.format(user))
        info = df.loc[user]
        avg = np.mean(list(info['rating']))
        std = np.std(list(info['rating']))

        return info, avg, std

    else:
        output = './raw_feature/user-user/{}/'.format(dataset)

        if not os.path.exists(output):
            os.makedirs(output)

        info_dict = {}
        for user in user_list:
            print('extracting user {}/{} information.......'.format(user+1, len(user_list)))
            info = df.loc[user]
            avg = np.mean(list(info['rating']))
            std = np.std(list(info['rating']))
            info_dict[user] = [avg, std]

        info_df = pd.DataFrame.from_dict(info_dict, orient='index')
        info_df.columns = ['mean', 'std']
        print(info_df.head())

        info_df.to_csv(output + '{}_user_info.csv'.format(dataset))


def get_fitted_dataset(df, dataset='train_0', user=None):
    """
    this code take fit user rating by using zscore
    it can be applied to a single certain user (user=user_id)
    or to all users in a given dataset and save result as a csv file (user=None)

    :param df:
    :param dataset:
    :param user:
    :return:
    """
    df = df.set_index('user_id')
    user_list = list(set(df.index))
    info = pd.read_csv('./raw_feature/{}/{}_user_info.csv'.format(dataset, dataset), sep=',', index_col=0)

    if user:

        mean = info['mean'].loc[user]
        std = info['std'].loc[user]
        df['rating'] = np.where(df.index==user, (df['rating']-mean)/std, df['rating'])

        return df.loc[user]

    else:
        output = './dataset/yahoo_r2_zscored/{}/'.format(dataset)

        if not os.path.exists(output):
            os.makedirs(output)

        for i in range(100):

            output_path = output + '{}_{}.csv'.format(dataset, str(i))
            if os.path.exists(output_path):
                print('part {}/100 already saved!!!!!!!'.format(i))
                continue

            start = i * len(user_list)//100
            end = start + len(user_list)//100

            for user in user_list[start:end]:
                print('zscoring user {}/{}, part {}/100.......'.format(user+1, len(user_list), i+1))
                mean = info['mean'].loc[user]
                std = info['std'].loc[user]
                df['rating'] = np.where(df.index == user, (df['rating'] - mean) / std, df['rating'])

            df.loc[user_list[start:end]].to_csv(output_path)


def fix_fitted_dataset(df, dataset='train_0'):
    """
    this code find users whoese fitted rating is NA (std = 0) and re-standardize
    using mean and std of the whole training set
    """
    df = df.set_index('user_id')
    avg = df['rating'].mean()
    std = df['rating'].std()
    output = './dataset/yahoo_r2_zscored/{}/'.format(dataset)
    print('dataset loaded')

    for i in range(100):

        path = output + '{}_{}.csv'.format(dataset, str(i))
        dfz = pd.read_csv(path, sep=',', index_col=0)
        user_list = set(dfz.index)

        for user in user_list:

            if math.isnan(dfz.loc[user]['rating'].iloc[0]):
                dfz.loc[user]['rating'] = (df.loc[user]['rating'] - avg) / std
                print('corrected user {}, {}/{}'.format(user, i+1, 100))
            # else:
            #     print('user {} does not need fix, {}/{}'.format(user, i+1, 100))

        dfz.to_csv(path)
        print('part {}/{} fixed'.format(i+1, 100))


def concat_fitted_dataset(dataset='train_0'):
    """
    this code combined split training set into one whole training set
    """
    output = './dataset/yahoo_r2_zscored/{}/'.format(dataset)
    df = pd.DataFrame()

    for i in range(100):

        path = output + '{}_{}.csv'.format(dataset, str(i))
        dfz = pd.read_csv(path, sep=',', index_col=0)
        df = pd.concat([df, dfz])
        print('concatenating part {}/{}'.format(i+1, 100))

    df.to_csv('./dataset/yahoo_r2_zscored/{}.csv'.format(dataset))


def item_similarity_matrix(path='./dataset/song-attributes.txt'):
    """
    this code calculate and save item similarity matrix using jaccard similarity
    """
    ct = load_content_info(path=path)
    item_list = list(ct.index)
    sim = np.zeros((len(item_list), len(item_list)))
    for i in range(len(item_list)):
        f1 = list(ct.loc[i])
        sim[i][i] = 1
        for j in range(i+1, len(item_list)):
            f2 = list(ct.loc[j])
            sim[i][j] = jaccard_similarity_score(f1, f2)
    np.save('./raw_feature/item_jaccard_sim.npy', sim)
    
# this is the main part
# choose to save info or fitted data
dataset = './dataset/yahoo_r2/train_0.txt'

#df = load_dataset(dataset)

# get_user_info(df, dataset='train_0', user=None)

# get_fitted_dataset(df, dataset='train_0', user=None)

# get_item_info(df, dataset='train_0')

# fix_fitted_dataset(df, dataset='train_0')

#concat_fitted_dataset(dataset='train_0')

item_similarity_matrix(path='./dataset/song-attributes.txt')