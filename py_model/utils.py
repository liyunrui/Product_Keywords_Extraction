#!/usr/bin/python3
# -*-coding:utf-8
'''
Created on Fri March 1 22:22:35 2017

@author: Ray

'''
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
import gc
from tqdm import tqdm


def mkdir_p(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def reduce_memory(df, ix_start=0):
    df.fillna(-1, inplace=True)
    df_ = df.sample(9999, random_state=71)
    ## int
    col_int8 = []
    col_int16 = []
    col_int32 = []
    for c in tqdm(df.columns[ix_start:], miniters=20):
        if df[c].dtype=='O':
            continue
        elif df[c].dtype == 'datetime64[ns]':
            continue  
        elif (df_[c] == df_[c].astype(np.int8)).all():
            col_int8.append(c)
        elif (df_[c] == df_[c].astype(np.int16)).all():
            col_int16.append(c)
        elif (df_[c] == df_[c].astype(np.int32)).all():
            col_int32.append(c)
    
    df[col_int8]  = df[col_int8].astype(np.int8)
    df[col_int16] = df[col_int16].astype(np.int16)
    df[col_int32] = df[col_int32].astype(np.int32)
    
    ## float
    col = [c for c in df.dtypes[df.dtypes==np.float64].index if '_id' not in c]
    df[col] = df[col].astype(np.float32)

    gc.collect()
