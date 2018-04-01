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
#-------------------------
# train
#-------------------------
'''
Basically, what I did in the below:
1.把有意義的字,用語意相近的字替換by human.
2.我們把不存在於word2vec的向量空間中的字,當作zero vector

'''
embedding_matrix = []
for w in train.words.tolist():
    if w in word_vectors.vocab:
        embedding_matrix.append(model.wv[w])
    elif re.match('^i\d', w):
        w = re.sub('^i\d', 'iphone', w)[:6]
        embedding_matrix.append(model.wv[w])		
    elif re.match('^oppo', w):
        w = 'oppo' 
        embedding_matrix.append(model.wv[w])             
    elif re.match('^iphone', w):
        w = re.sub('^iphone', 'iphone', w)[:6]
        embedding_matrix.append(model.wv[w])
    elif re.search('mah', w):
        w = 'mah'
        embedding_matrix.append(model.wv[w])
    elif re.search('note', w):
        w = 'note' 
        embedding_matrix.append(model.wv[w])
    elif re.search('t恤', w):
        w = '恤'     
        embedding_matrix.append(model.wv[w])
    elif re.search('男生衣著', w):
        w = '男裝'                        
        embedding_matrix.append(model.wv[w])
    elif re.search('女生衣著', w):
        w = '女裝'  
        embedding_matrix.append(model.wv[w])
    elif re.search('行動電源', w):
        w = '充電器'                         
        embedding_matrix.append(model.wv[w])           
    elif re.search('大尺碼', w):
        w = '尺碼'   
        embedding_matrix.append(model.wv[w])
    elif re.search('連身褲', w):
        w = '褲'                            
        embedding_matrix.append(model.wv[w])                    
    elif re.search('寬褲', w):
        w = '褲'                            
        embedding_matrix.append(model.wv[w])                    
    elif re.search('破褲', w):
        w = '褲'   
        embedding_matrix.append(model.wv[w])
    elif re.search('藍芽耳機', w):
        w = '藍芽'             
        embedding_matrix.append(model.wv[w])
    elif re.search('手機殼', w):
        w = '機殼'             
        embedding_matrix.append(model.wv[w])
    elif re.search('一字領', w):
        w = '字領'             
        embedding_matrix.append(model.wv[w])
    elif re.search('充電線', w):
        w = '充電器'             
        embedding_matrix.append(model.wv[w])
    elif re.search('手機殼', w):
        w = '機殼'             
        embedding_matrix.append(model.wv[w])
    elif re.search('長洋裝', w):
        w = '洋裝'             
        embedding_matrix.append(model.wv[w])
    elif re.search('保護貼', w):
        w = '貼膜'     
        embedding_matrix.append(model.wv[w])
    elif re.search('type c', w):
        w = 'usb'                      
        embedding_matrix.append(model.wv[w])
    elif re.search('男套裝', w):
        w = '運動裝'                      
        embedding_matrix.append(model.wv[w])
    elif re.search('熱銷批', w):
        w = '熱銷'                      
        embedding_matrix.append(model.wv[w])
    elif re.search('蝦幣', w):
        w = '貨幣'                      
        embedding_matrix.append(model.wv[w])
    elif re.search('运动套装', w):
        w = '運動裝'                      
        embedding_matrix.append(model.wv[w])
    elif re.search('孕媽', w):
        w = '孕婦'                      
        embedding_matrix.append(model.wv[w])
    elif re.search('美少女戰士', w):
        w = '美少女'                      
        embedding_matrix.append(model.wv[w])
    elif re.search('棒球外套', w):
        w = '外套'                      
        embedding_matrix.append(model.wv[w])
    elif re.search('大學t', w):
        w = '恤'               
        embedding_matrix.append(model.wv[w])
    elif re.search('隨身碟', w):
        w = 'usb'                     
        embedding_matrix.append(model.wv[w])
    elif re.search('傳說對決', w):
        w = '手遊'
        embedding_matrix.append(model.wv[w])
    elif re.search('gymshark', w):
        w = '運動服裝'       
        embedding_matrix.append(model.wv[w])
    elif re.search('ua', w):
        w = '運動服裝'    
        embedding_matrix.append(model.wv[w])
    elif re.search('under armour', w):
        w = '運動服裝'
        embedding_matrix.append(model.wv[w])        
    elif re.search('飛天小女警', w):
        w = '小女警'                         
        embedding_matrix.append(model.wv[w])                     
    else:
    	embedding_matrix.append(np.zeros(250))

embedding_matrix = pd.DataFrame(np.array(embedding_matrix))
embedding_matrix.columns = ['dim_{}'.format(i+1) for i in range(250)]

train = pd.concat([train,embedding_matrix], axis = 1)
train.drop(['ratio_of_keywords','Category'],axis = 1, inplace = True)

#-------------------------
# test
#-------------------------
embedding_matrix = []
for w in test.words.tolist():
    if w in word_vectors.vocab:
        embedding_matrix.append(model.wv[w])
    elif re.match('^i\d', w):
        w = re.sub('^i\d', 'iphone', w)[:6]
        embedding_matrix.append(model.wv[w])
    elif re.match('^oppo', w):
        w = 'oppo' 
        embedding_matrix.append(model.wv[w])             
    elif re.match('^iphone', w):
        w = re.sub('^iphone', 'iphone', w)[:6]
        embedding_matrix.append(model.wv[w])
    elif re.search('mah', w):
        w = 'mah'
        embedding_matrix.append(model.wv[w])
    elif re.search('note', w):
        w = 'note' 
        embedding_matrix.append(model.wv[w])
    elif re.search('t恤', w):
        w = '恤'     
        embedding_matrix.append(model.wv[w])
    elif re.search('男生衣著', w):
        w = '男裝'                        
        embedding_matrix.append(model.wv[w])
    elif re.search('女生衣著', w):
        w = '女裝'  
        embedding_matrix.append(model.wv[w])
    elif re.search('行動電源', w):
        w = '充電器'                         
        embedding_matrix.append(model.wv[w])           
    elif re.search('大尺碼', w):
        w = '尺碼'   
        embedding_matrix.append(model.wv[w])
    elif re.search('連身褲', w):
        w = '褲'                            
        embedding_matrix.append(model.wv[w])                    
    elif re.search('寬褲', w):
        w = '褲'                            
        embedding_matrix.append(model.wv[w])                    
    elif re.search('破褲', w):
        w = '褲'   
        embedding_matrix.append(model.wv[w])
    elif re.search('藍芽耳機', w):
        w = '藍芽'             
        embedding_matrix.append(model.wv[w])
    elif re.search('手機殼', w):
        w = '機殼'             
        embedding_matrix.append(model.wv[w])
    elif re.search('一字領', w):
        w = '字領'             
        embedding_matrix.append(model.wv[w])
    elif re.search('充電線', w):
        w = '充電器'             
        embedding_matrix.append(model.wv[w])
    elif re.search('手機殼', w):
        w = '機殼'             
        embedding_matrix.append(model.wv[w])
    elif re.search('長洋裝', w):
        w = '洋裝'             
        embedding_matrix.append(model.wv[w])
    elif re.search('保護貼', w):
        w = '貼膜'     
        embedding_matrix.append(model.wv[w])
    elif re.search('type c', w):
        w = 'usb'                      
        embedding_matrix.append(model.wv[w])
    elif re.search('男套裝', w):
        w = '運動裝'                      
        embedding_matrix.append(model.wv[w])
    elif re.search('熱銷批', w):
        w = '熱銷'                      
        embedding_matrix.append(model.wv[w])
    elif re.search('蝦幣', w):
        w = '貨幣'                      
        embedding_matrix.append(model.wv[w])
    elif re.search('运动套装', w):
        w = '運動裝'                      
        embedding_matrix.append(model.wv[w])
    elif re.search('孕媽', w):
        w = '孕婦'                      
        embedding_matrix.append(model.wv[w])
    elif re.search('美少女戰士', w):
        w = '美少女'                      
        embedding_matrix.append(model.wv[w])
    elif re.search('棒球外套', w):
        w = '外套'                      
        embedding_matrix.append(model.wv[w])
    elif re.search('大學t', w):
        w = '恤'               
        embedding_matrix.append(model.wv[w])
    elif re.search('隨身碟', w):
        w = 'usb'                     
        embedding_matrix.append(model.wv[w])
    elif re.search('傳說對決', w):
        w = '手遊'
        embedding_matrix.append(model.wv[w])
    elif re.search('gymshark', w):
        w = '運動服裝'      
        embedding_matrix.append(model.wv[w])
    elif re.search('ua', w):
        w = '運動服裝'    
        embedding_matrix.append(model.wv[w])
    elif re.search('under armour', w):
        w = '運動服裝'
        embedding_matrix.append(model.wv[w])
    elif re.search('飛天小女警', w):
        w = '小女警'                         
        embedding_matrix.append(model.wv[w])                     
    else:
    	embedding_matrix.append(np.zeros(250))

embedding_matrix = pd.DataFrame(np.array(embedding_matrix))
embedding_matrix.columns = ['dim_{}'.format(i+1) for i in range(250)]

test = pd.concat([test,embedding_matrix], axis = 1)
test.drop('Category',axis = 1, inplace = True)

#-------------------------
#save
#-------------------------

train.to_csv('../feature/{}/word_vec.csv.gz'.format('train'), index = False, compression='gzip')
test.to_csv('../feature/{}/word_vec.csv.gz'.format('test'), index = False, compression='gzip')

