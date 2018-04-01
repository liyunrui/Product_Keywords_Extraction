# -*- coding: utf8
import pandas as pd
import numpy as np
from gensim.models import word2vec
from gensim import models
import jieba
import jieba.analyse
import warnings
warnings.filterwarnings("ignore")
import re



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

# model trained by word2vec
model = models.Word2Vec.load('../word2vec/word2vec.model')
word_vectors =  model.wv # for checking a word exist in model
print ('loading done')
unseen_words = []
for w in train.words.unique():
    if w in word_vectors.vocab:
        model.wv[w].shape
    else:
        #1. 把有意義的字替換成為相近語意的字by human
        if re.match('^i\d', w):
            w = re.sub('^i\d', 'iphone', w)[:6]
        elif re.match('^oppo', w):
            w = 'oppo'              
        elif re.match('^iphone', w):
            w = re.sub('^iphone', 'iphone', w)[:6]
        elif re.search('mah', w):
            w = 'mah'
        elif re.search('note', w):
            w = 'note' 
        elif re.search('t恤', w):
            w = '恤'     
        elif re.search('男生衣著', w):
            w = '男裝'                        
        elif re.search('女生衣著', w):
            w = '女裝'  
        elif re.search('行動電源', w):
            w = '充電器'                                    
        elif re.search('大尺碼', w):
            w = '尺碼'   
        elif re.search('連身褲', w):
            w = '褲'                                                
        elif re.search('寬褲', w):
            w = '褲'                                                
        elif re.search('破褲', w):
            w = '褲'   
        elif re.search('藍芽耳機', w):
            w = '藍芽'             
        elif re.search('手機殼', w):
            w = '機殼'             
        elif re.search('一字領', w):
            w = '字領'             
        elif re.search('充電線', w):
            w = '充電器'             
        elif re.search('手機殼', w):
            w = '機殼'             
        elif re.search('長洋裝', w):
            w = '洋裝'             
        elif re.search('保護貼', w):
            w = '貼膜'     
        elif re.search('type c', w):
            w = 'usb'                      
        elif re.search('男套裝', w):
            w = '運動裝'                      
        elif re.search('熱銷批', w):
            w = '熱銷'                      
        elif re.search('蝦幣', w):
            w = '貨幣'                      
        elif re.search('运动套装', w):
            w = '運動裝'                      
        elif re.search('孕媽', w):
            w = '孕婦'                      
        elif re.search('美少女戰士', w):
            w = '卡通'                      
        elif re.search('棒球外套', w):
            w = '外套'                      
        elif re.search('大學t', w):
            w = '恤'               
        elif re.search('隨身碟', w):
            w = 'usb'                     
        elif re.search('傳說對決', w):
            w = '手遊'
        elif re.search('gymshark', w):
            w = '運動服裝'       
        elif re.search('飛天小女警', w):
            w = '卡通'                                              
        else:
            unseen_words.append(w)
#--------------------------            
# save those unseen words
#--------------------------
df = pd.DataFrame({'unseen_words':unseen_words})
df.to_csv('../input/unseen_words.csv',index = False)
