# -*- coding: utf8
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import utils # made by author for efficiently dealing with data
import warnings
warnings.filterwarnings("ignore")

##################################
# loading data
##################################

df0 = pd.read_csv('../input/train_mobile.csv.gz', compression='gzip')
df1 = pd.read_csv('../input/train_female_fashion.csv.gz', compression='gzip')
df2 = pd.read_csv('../input/train_male_fashion.csv.gz', compression='gzip')
train = pd.concat([df0, df1, df2], ignore_index=True)

df0 = pd.read_csv('../input/test_mobile.csv.gz', compression='gzip')
df1 = pd.read_csv('../input/test_female_fashion.csv.gz', compression='gzip')
df2 = pd.read_csv('../input/test_male_fashion.csv.gz', compression='gzip')
test = pd.concat([df0, df1, df2], ignore_index=True)


def len_extracted_words(x):
    '''
    parameters:
    x: str
    '''
    return len(list(set(list(jieba.analyse.extract_tags(x)))))
def position_given_product_name(x):
    '''
    斷詞後, 此word在於product_name中的位置
    parameters:
    --------------
    x:DataFrame
    '''
    seg = list(set(list(jieba.cut_for_search(x['Product Name']))))
    if x['words'] in set(seg):
        x['position_given_product_name'] = 1.0 * seg.index(x['words']) / len(seg)
    else:
        x['position_given_product_name'] = -1.0
    return x
def weights_given_product_name(x):
    '''
    
    parameters:
    -------------
    x: DataFrame
    x['Product Name'].unique()[0]: str
    '''    
    extracted_w = dict(jieba.analyse.extract_tags(x['Product Name'].unique()[0], withWeight = True, topK = len(x['Product Name'].unique()[0])))
    x['weights_given_product_name'] = [extracted_w[w] if w in extracted_w else 0.0 for w in x['words']]
    return x
#-------------------------
# train
#-------------------------
col = ['Product Name','words','weights_given_product_name','position_given_product_name','len_extracted_words']
train['len_extracted_words'] = train['Product Name'].apply(len_extracted_words)
train = train.apply(position_given_product_name, axis = 1)
train = train.groupby('Product Name').apply(weights_given_product_name)[col]

#-------------------------
# test
#-------------------------
col = ['Product Name','words','weights_given_product_name','position_given_product_name','len_extracted_words']
test['len_extracted_words'] = test['Product Name'].apply(len_extracted_words)
test = test.apply(position_given_product_name, axis = 1)
test = test.groupby('Product Name').apply(weights_given_product_name)[col]

#-------------------------
#save
#-------------------------
utils.mkdir_p('../feature/train')
utils.mkdir_p('../feature/test')

train.to_csv('../feature/{}/relationship_between_word_and_productname_mobile.csv.gz'.format('train'), index = False, compression='gzip')
test.to_csv('../feature/{}/relationship_between_word_and_productname_mobile.csv.gz'.format('test'), index = False, compression='gzip')

