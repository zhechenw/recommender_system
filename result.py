#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:16:59 2019

@author: zhechenw
"""

import pandas as pd
from data_processing import *


test_path = './dataset/test_0.txt'


test_df = load_dataset(test_path)

train_path = './dataset/train_0.txt'
contents_path = './dataset/song-attributes.txt'


train_df = load_dataset(train_path)
item_content_df = load_item_contents(contents_path)

U = Users(train_df)
u_list = U.u_list
I = Items(item_content_df, train_df)
