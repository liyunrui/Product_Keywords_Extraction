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


def is_word_brand_name(x):
    '''
    create a binary feautre, 1:is a brande name, otherwise 0.
    
    parameters
    -------------------
    x : str
    '''
    pattern = [
 'mercci22',
 'a7',
 'plus',
 'skinnydip',
 'sony c3',
 'oppo r9s',
 'sony z5p',
 'samsung s6',
 'eugene tong',
 'samsung j7',
 'j3',
 'af',
 'oppo a57',
 'nokia 6',
 'iphone 5',
 'htc 828',
 'calvin klein',
 'sony xa',
 's5',
 'scottish house',
 's4',
 'htc x9',
 'e7',
 'sony c5',
 'iphone6s',
 'kenzo',
 'hush puppies',
 'air space',
 'carhartt',
 'nokia',
 'iphone 7s',
 'r9',
 'sony xz',
 '64g',
 'voda swim',
 'champion',
 'ipad mini',
 'oppo',
 'prettynice',
 '6s plus',
 'richmond&finch',
 'stussy',
 'adidas',
 'ryan',
 'sony xa1',
 'samsung',
 'mate8',
 'j5',
 'bikini',
 's7 edge',
 'oppo f1f',
 'anden hud',
 'i7plus',
 'iphone 7',
 'a5',
 'htc728',
 'a8',
 'ipad air',
 'samsung s8 plus',
 '16g',
 'type c',
 'oppo f1s',
 'sharp z2',
 'queen shop',
 'moschino',
 'iphone5s',
 'fila',
 'g5',
 'samsung c9 pro',
 '4ds studio',
 't shirt',
 'h&m',
 'u11 htc',
 'brandy melville',
 'anti social',
 's6',
 'gap',
 'ipad pro',
 'j7 pro',
 'm8',
 'oppo a77',
 'r11 android',
 'i6s plus',
 'ipad 9.7',
 'oppo r11',
 'oppo r9s plus',
 'oversize',
 's8 plus',
 'iphone 6s',
 'note6',
 'ripndip',
 'edge',
 'fendi',
 'lolita',
 'timberland',
 'n3',
 'xa1 ultra',
 'dickies',
 'iphone6s plus',
 'note7',
 'zenfone go',
 'j7 prime',
 'note5',
 'r7',
 'kyrie irving',
 'j7',
 'pazzo',
 'uniqlo',
 'number nine',
 'sony z5',
 'a3',
 'skinny dip',
 'iphone 7 plus',
 'r5',
 'kurbis store',
 'iphone 6',
 'ralph lauren',
 'samsung a8',
 'interbreed',
 's7',
 'karl lagerfeld',
 'roxy',
 'oppo f1',
 'h connect',
 'htc 10 evo',
 'tommy hilfiger',
 'iphone 5s',
 'p9',
 'oppo a39',
 'roots',
 'ih nom uh nit',
 'jabra',
 'zenfone 3',
 'marjorie',
 'madness',
 'adidas originals',
 'note2',
 'bts',
 'note3',
 'sony m4',
 's3',
 'garmin 235',
 'ua',
 'oppo r9',
 'a la sha',
 'note4',
 'zara',
 'htc a9',
 'c9 pro',
 'hollister',
 '32g',
 'm2 plus',
 'zenfone selfie',
 'iphone se',
 'f1',
 'iphone7',
 'iphone7s',
 'iphone5',
 'vlone',
 'g4',
 'iphone6',
          ]
    label = None
    if x in set(pattern):
        label = 1
    else:
        label = 0
    return label

#-------------------------
# train
#-------------------------
col = ['Product Name','words','is_word_brand_name']
train['is_word_brand_name'] = train['words'].apply(is_word_brand_name)
train = train[col]

#-------------------------
# test
#-------------------------
col = ['Product Name','words','is_word_brand_name']
test['is_word_brand_name'] = test['words'].apply(is_word_brand_name)
test = test[col]

#-------------------------
#save
#-------------------------

train.to_csv('../feature/{}/is_word_brand_name.csv.gz'.format('train'), index = False, compression='gzip')
test.to_csv('../feature/{}/is_word_brand_name.csv.gz'.format('test'), index = False, compression='gzip')

