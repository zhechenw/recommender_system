#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 20:49:05 2019

@author: zhechenw
"""
import math, json
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, kendalltau
from scipy.spatial.distance import euclidean, cosine, correlation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import jaccard_score


def load_inb(u_id):
    with open('./initial_nb/inb{}.json'.format(u_id), 'r') as f:
        inb = json.load(f)
    
    if u_id <= 1487:
        return sorted(inb, key=lambda x: len(x[1][0]), reverse=True)
    else:
        return inb


def load_item_sim(i_id1, i_id2):
    if i_id1 != i_id2:
        i_id1, i_id2 = min(i_id1, i_id2), max(i_id1, i_id2)
        
    else:
        return 1
        
    with open('./item_sim/{}.json'.format(i_id1), 'r') as f:
        sim = json.load(f)
        
    return sim[i_id2]

    
def get_similarity(u_id, limit=100, sim_fn=kendalltau):
    sim_map = {}
    inb = load_inb(u_id)[:limit]
    for i in range(len(inb)):
        inb_id = inb[i][0]
        f1 = inb[i][1][0]
        f2 = inb[i][1][1]
        
        if sim_fn == (cosine or euclidean):
            sim = 1 - sim_fn(f1, f2)
        if sim_fn == (kendalltau or pearsonr):
            sim = sim_fn(f1, f2)[0]
        
        if math.isnan(sim):
            sim_map[inb_id] = 0
        else:
            sim_map[inb_id] = sim
        
    return sim_map


def get_knn(u_id, k=6, limit=100, sim_fn=kendalltau):
    sim = get_similarity(u_id, limit=limit, sim_fn=sim_fn)
    sorted_sim = sorted(sim.items(), key=lambda x: x[1], reverse=True)
    knn = {}
    for i in range(k):
        knn[sorted_sim[i][0]] = sorted_sim[i][1]
        
    return knn
    
    
def get_UCF_score(u_id, U, k=6, limit=100, sim_fn=kendalltau):
    knn = get_knn(u_id, k=k, limit=limit, sim_fn=sim_fn)
    UCF_score = {}
    u_i_list = U.get_user(u_id).i_list
    for u, s in tqdm(knn.items()):
        u_temp = U.get_user(u)
        u_record = u_temp.record
        for song in list(u_temp.i_list):
            if song in u_i_list:
                continue
            
            if song in UCF_score.keys():
                with np.errstate(divide='raise'):
                    try:
                        UCF_score[song] += s * (u_record[song] - u_temp.rating_avg) / u_temp.rating_std
                    except FloatingPointError:
                        UCF_score[song] += s * (u_record[song] - U.rating_avg) / U.rating_std
            else:
                with np.errstate(divide='raise'):
                    try:
                        UCF_score[song] = s * (u_record[song] - u_temp.rating_avg) / u_temp.rating_std
                    except FloatingPointError:
                        UCF_score[song] = s * (u_record[song] - U.rating_avg) / U.rating_std
    
    return UCF_score


def get_candidates(UCF_score):
    return list(UCF_score.keys())
        
        
def get_CF_score(u_id, candidates, U, I, sim_fn=jaccard_score):
    i_list = U.get_user(u_id).i_list
    CF_score = {}
    for i_id1 in tqdm(candidates):
        f1 = I.get_item(i_id1).contents
        sim = 0
        for i_id2 in i_list:
            try:
                sim += load_item_sim(i_id1, i_id2)
            except:
                f2 = I.get_item(i_id2).contents
                sim += sim_fn(f1, f2, average='micro')
        CF_score[i_id1] = sim / len(i_list)
    
    return CF_score


def get_POP_score(u_id, candidates, I):
    pop_map = I.pop_map
    POP_score = {}
    for i_id in tqdm(candidates):
        POP_score[i_id] = pop_map[i_id]
        
    return POP_score
        

def get_final_score(UCF, CF, POP, weights=None):
    score = np.array((list(UCF.values()), list(CF.values()), list(POP.values()))).T
    score = StandardScaler().fit_transform(score)
    if weights:
        return score.dot(np.array(weights).reshape(3,1))
    else:
        return score


def load_scores(u_id, k, sim_fn):
    
    score_path = './scores/{}_k{}_{}.npy'.format(sim_fn, k, u_id)
    return np.load(score_path)


def recommender(u_id, k, sim_fn, weights, n=None):
    
    if sim_fn not in ['cos', 'ken']:
        raise ValueError('sim_fn has to be either string cos or ken')
    
    try:
        score = load_scores(u_id, k, sim_fn)
        i_list = score[0]
        scores = np.delete(score, 0, 0).T.dot(weights)
        score_map = {int(k):v for k, v in dict(np.array((i_list, scores)).T).items()}
        
        return dict(sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:n])
        
    except:
        raise ValueError('loading scores failed, check score file!') 


class Trainer:
    def __init__(self, U, n=None):

        self.inb_map = {}
        self.u_list = U.u_list[:n]
        self.U = U
        
        for i in range(len(self.u_list)):
            u1 = U.get_user(self.u_list[i])
            inb_temp = {}
            
            for j in range(i+1, len(self.u_list)):
                u2 = U.get_user(self.u_list[j])
                intersect = np.intersect1d(u1.i_list, u2.i_list, assume_unique=True)
                f1, f2 = [], []
                
                if len(intersect) > 2: # kendaltau raise error when tersect <= 2
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
                    
                    inb_temp[u2.u_id] = (f1, f2)
            
            sorted_inb = sorted(inb_temp.items(), key=lambda x: len(x[1]), reverse=True)
            self.inb_map[u1.u_id] = sorted_inb
        
        
    def get_inb(self, u_id, limit=100):
        return self.inb_map[u_id][:limit]
    
