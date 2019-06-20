#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 00:52:42 2019

@author: zhechenw
"""

import json, os
from data_processing import *
from sklearn.metrics import jaccard_score


train_path = './dataset/train_0.txt'
contents_path = './dataset/song-attributes.txt'

train_df = load_dataset(train_path)
item_content_df = load_item_contents(contents_path)

I = Items(item_content_df, train_df)

for i in range(len(I.i_list)):
    i_id1 = I.i_list[i]
    output = './item_sim/{}.jason'.format(i_id1)
    if os.path.exists(output):
        print('item {} similarity already saved!'.format(i_id1))
        continue
    
    f1 = I.get_item(i_id1).contents

    sim = {}
    
    for j in range(i+1, len(I.i_list)):
        i_id2 = I.i_list[j]
        f2 = I.get_item(i_id2).contents
        sim[i_id2] = jaccard_score(f1, f2, average='micro')
        
    with open(output, 'w') as f:
        json.dump(sim, f)
    print('item {} similarity saved!'.format(i_id1))
