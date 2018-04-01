#!/usr/bin/python3
# -*-coding:utf-8
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import utils # written by author
import multiprocessing as mp
import gc # for automatic releasing memory

def word_given_product_name_feature(df, name):
    #relationship_between_word_and_productname_mobile
    df = pd.merge(df, pd.read_csv('../feature/{}/relationship_between_word_and_productname_mobile.csv.gz'.format(name), compression='gzip'),
                  on=['Product Name','words'], how='left')
    return df
def semantic_feature(df, name):
    #word_vec
    df = pd.merge(df, pd.read_csv('../feature/{}/word_vec.csv.gz'.format(name), compression='gzip'),
                 on=['Product Name','words'], how='left')
    gc.collect()
    return df
def hot_search_count(df, name):
    '''
    Here's feature will be updated as time goes on. 
    The reason is the words that user is likely to search is continuously changed with trends.
    '''
    #hot_search_count
    df = pd.merge(df, pd.read_csv('../feature/{}/hot_search_count.csv.gz'.format(name), compression='gzip'),
                 on=['Product Name','words'], how='left')
    #is_word_brand_name
    df = pd.merge(df, pd.read_csv('../feature/{}/is_word_brand_name.csv.gz'.format(name), compression='gzip'),
                  on=['Product Name','words'], how='left')

    gc.collect()
    return df


def concat_pred_features(T):
    if T == -1:
        name = 'test'
        # concat
        df0 = pd.read_csv('../input/test_mobile.csv.gz', compression='gzip')
        df1 = pd.read_csv('../input/test_female_fashion.csv.gz', compression='gzip')
        df2 = pd.read_csv('../input/test_male_fashion.csv.gz', compression='gzip')
        df = pd.concat([df0, df1, df2], ignore_index=True)
    else:
        name = 'train'
        #==============================================================================
        print('load label')
        #==============================================================================        
        # concat
        df0 = pd.read_csv('../input/train_mobile.csv.gz', compression='gzip')
        df1 = pd.read_csv('../input/train_female_fashion.csv.gz', compression='gzip')
        df2 = pd.read_csv('../input/train_male_fashion.csv.gz', compression='gzip')
        df = pd.concat([df0, df1, df2], ignore_index=True)

    
    #==============================================================================
    print('word_given_product_name feature')
    #==============================================================================
    df = word_given_product_name_feature(df, name)
    
    print('{}.shape: {}'.format(name, df.shape))

    #==============================================================================
    print('semantic_feature')
    #==============================================================================
    df = semantic_feature(df, name)
    
    print('{}.shape: {}'.format(name, df.shape))

    #==============================================================================
    print('hot search feature')
    #==============================================================================
    df = hot_search_count(df, name)
    
    print('{}.shape: {}'.format(name, df.shape))

    #==============================================================================
    print('feature engineering')
    #==============================================================================
    df = pd.get_dummies(df, columns=['Category'])
    df.drop_duplicates(['Product Name', 'words'], inplace = True)
    print('{}.shape: {}'.format(name, df.shape))   
    # some features with largest, we perform log transformation to them
    for col in df.columns:
        if 'count' in col and df[col].max() > 100:
            df['log_{}'.format(col)] = np.log(df[col] + 1) # smoothing
            df.drop(col, axis = 1, inplace = True)
    if name == 'train':
        #==============================================================================
        print('reduce memory')
        #==============================================================================
        utils.reduce_memory(df)
    #==============================================================================
    print('output')
    #==============================================================================
    df.to_csv('../feature/{}/all_features.csv.gz'.format(name), index = False, compression='gzip')

def multi(name):
    concat_pred_features(name)

##################################################
# Main
##################################################
s = time.time()

mp_pool = mp.Pool(2)
mp_pool.map(multi, [1, -1])

e = time.time()
print (e-s)
