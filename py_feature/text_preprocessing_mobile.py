# -*- coding: utf8
import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import re
import warnings
warnings.filterwarnings("ignore")


###########################
#loading data
###########################
'''
Product Name: 此user用某字搜尋後,出現在網頁中的其中一個商品名字
Category:分類
Query: 此user在shopee網站上用來query某商品下的關鍵字
Event: 代表此user在搜尋某一個關鍵字之後, 是否有點下該Product Name所在的網頁。Click if yes else Impression
date: 此user搜尋的時間

'''
# sample is snippet of view log that contains user behavior
sample = pd.read_csv('../raw_data/sample_data.csv',header = None)
sample.columns = ['Product Name','Category','Query','Event','Date']
# test: it contains 200 Proudct Name. we should find the best keywords of each product name. 

test = pd.read_csv('../raw_data/test_data.csv')

# remove outlier
sample.drop(sample.index[5139], inplace = True)
sample = sample.head(n = 10303)
# remove duplicated testing data
test.drop_duplicates(['Product Name','Category'], inplace = True)
test = test.reset_index().drop('index', axis = 1)

###########################
# Category: Mobile_Gadgets
###########################
train = sample[sample.Category == 'Mobile & Gadgets']

test = test[test.Category == 'Mobile & Gadgets']
test = test.reset_index().drop('index', axis = 1)

def extracted_keywords_by_jeiba(x):
    '''
    Convert Product name to tokens
    parameters:
    --------------
    x: str
    '''
    #1. Normalizing相同意義的字
    x = x.replace('i phone','iphone')
    x = x.replace('藍牙耳機','藍芽耳機')
    x = x.replace('保護套','手機殼')
    #2. 添加自定义词
    for w in ['行動電源','太陽能','android保護殼','iphone保護殼','手機殼','細圖',
              'android充電傳輸','iphone充電傳輸','客制化','書本套','智障手機',
              'type c','快充線','卡娜赫拉','充電線','安卓','大理石','達菲','王者榮耀',
              '華碩','紅米note3','sony xa','踢不爛','米奇','無臉男','手機架','懶人支架',
              '手機支架','快充線','卡娜赫拉','充電線','安卓','汽車手機架','熊熊遇見你',
              '手機貼紙','華碩','紅米note3','防摔','手機殼','廣角鏡頭','零件機','自拍棒',
              '灌籃高手','北極熊','美少女戰士','傳說對決','吸盤搖桿','美少女戰士','網路卡',
              '網卡','展示機','老人機','玻璃貼','藍芽耳機','儲值卡','保護貼','紅米4x','小米max2',
              '紅米note4','三星j730','無線充電','愛迪達','耐及','紅米note4x','小熊維尼','隨身碟',
              '火烈鳥','小米 max2','寶可夢','氣囊支架','飛天小女警','客製化','貓咪大戰爭','小飛象',
              '紅米note4x','三星','華為','軟殼','運動耳機','空機','嚕嚕米','螢幕','手機套','防塵塞',
              '情侶','史迪奇','潮牌','電池',
             ]:
        jieba.add_word(w)
    # 加载自定义词库
    #jieba.load_userdict("../input/customized_dict_for_Mobile_Gadgets.txt")
    tokens = None
    if len(jieba.analyse.extract_tags(x, topK = len(x))) != 0:
        tokens = jieba.analyse.extract_tags(x, topK = len(x))
    else:
        tokens = [x]
    # remove stopwords
    return list(set(tokens))


def is_querytokens_in_product_nametokens(x):
    '''
    
    parameters:
    --------------
    x: DataFrame
    '''
    label = None
    if set(x['query_tokens']) & set(x['product_name_tokens']):
        label = 1
    else:
        label = 0
    return label

def get_new_token(list1, list2):
    # list2: longer tokens
    # list1: shorer tokens
    empty_list = list()
    for k in set(list1).intersection(set(list2)):
        if list(set(list2).difference(set(list1))) == empty_list:
            continue
        elif k in list(set(list2).difference(set(list1)))[0]:
            list2.remove(k)
    return list2   
def remove_stopwords(tokens):
    '''
    Remove stopwords(the tokenized words from Query, buy never be searched by user)

    parameters:
    --------------
    tokens: list
    '''
    for s_w in stop_words:
        if s_w in tokens and len(tokens) != 1:
            tokens.remove(s_w) # if len(tokens == 1, 就不removed, avoiding empty list)
    return tokens
###########################
# Text preprocessing
###########################

# covert word to lower()
train['Query'] = [q.lower() for q in train['Query']]
train['Product Name'] = [p_n.lower() for p_n in train['Product Name']]

# okenization
train['product_name_tokens'] = train['Product Name'].apply(extracted_keywords_by_jeiba)
train['query_tokens'] = train['Query'].apply(extracted_keywords_by_jeiba)

# create stopwords, which are defined by the words that are exactly used to search prodcut by users.
stop_words = []
for ix, row in train.iterrows():
    for q_t in row.query_tokens:
        if q_t not in train.Query.tolist():
            # 我要query的tokens準確是使用者會搜尋的字
            stop_words.append(q_t)
stop_words = list(set(stop_words))
# remove stopwords
train['product_name_tokens'] = train['product_name_tokens'].apply(remove_stopwords)
train['query_tokens'] = train['query_tokens'].apply(remove_stopwords)

#--------------
# 針對特殊pattern such as brand name 做處理, for making sure get a better tokenizing result.
#--------------
pattern = ['type c','oppo r11','sony xa','sony xz','oppo r9','sony c3','iphone 7','ipad pro',
          'richmond&finch','xa1 ultra','m2 plus','s7 edge','skinny dip','s7 edge','j7 prime',
          'sony m4','sony z5','i6s plus','samsung j7','htc728','iphone se','s7 edge',
           'oppo f1s','ipad pro','iphone 7 plus','samsung c9 pro','sony xa1','zenfone selfie',
           'samsung s8 plus','oppo a57','i6s plus','ipad air','nokia 6','u11 htc','c9 pro',
           'oppo f1f','samsung a8','j7 prime','sharp z2','oppo r9s','iphone 7 plus',
           'samsung s6','oppo r9s plus','s7 edge','zenfone 3','ipad mini','oppo f1s','oppo a77',
           'c9 pro','j7 pro','nokia 6','oppo a39','6s plus','garmin 235','htc 10 evo','htc a9','iphone6s plus',
           'r11 android','sony z5p','ipad 9.7','sony c5','sony m4','oppo f1','zenfone go','nokia','htc 828',
           'htc x9','s8 plus','4ds studio','i7plus','jabra','samsung','a8','oppo','j5','a7','a5','a3',
           's3','s4','s5','s6','s7','edge','note2','note3','note4','note5','note6','a8','e7','j3','j5','j7',
           'g4','g5','r5', 'r7', 'r9', 'plus', 'f1', 'n3','p9', 'plus', 'mate8','iphone5', 'iphone6', 'iphone7',
           'iphone 5', 'iphone 6', 'iphone 7','iphone 5s', 'iphone 6s', 'iphone 7s','iphone5s', 'iphone6s', 'iphone7s',
           'nokia','note3', 'note4', 'note5', 'note6', 'note7', '16g','32g','64g','skinnydip',
           'sony xz','oppo r9','fendi','kenzo','ryan','m8','xperia xz','htc one e9 dual sim',
           'iphone7 plus','iphone6 plus','iphone5 plus','iphone4 plus','iphone8 plus'
          ] # for English word
pattern.sort(key = lambda s: len(s)) # sorted by length of str for handling wiht problem: 'oppo r9' and 'oppo r9s'
for ix, row in train[['Product Name','Query','product_name_tokens','query_tokens']].iterrows():
   # 手動處理特殊pattern的分詞
    for s_p in pattern:
        # text preprocessing for query
        if s_p in row['Query']:
            # replace query_tokesn with new_tokens if special pattern in query
            train.query_tokens.ix[ix] = list(set(get_new_token(row['query_tokens'], row['query_tokens'] + re.findall(s_p, row['Query']))))
        # text preprocessing for Product Name
        if s_p in row['Product Name']:
            # replace query_tokesn with new_tokens if special pattern in query
            train.product_name_tokens.ix[ix] = list(set(get_new_token(row['product_name_tokens'], row['product_name_tokens'] + re.findall(s_p, row['Product Name']))))

# Creat Label
train['is_querytokens_in_product_nametokens'] = train.apply(is_querytokens_in_product_nametokens, axis = 1)
train.reset_index(drop = True, inplace = True)
#-------------------------
# train
#-------------------------

output = ['Product Name','product_name_tokens','Category','query_tokens']
word_list = []
is_keyword = []
product_name = []
category = []
for ix, row in train[train.is_querytokens_in_product_nametokens == 1][output].iterrows():
    # row['product_name_tokens']: 藉由jeiba設計的客製化Product Keyword Extraction function, 所產生的關鍵字
    # row['query_tokens']: the words(tokens) that customer used to look for product
    for w in row['product_name_tokens']:
        label = None
        if w in set(row['query_tokens']):
            label = 1
        else:
            label = 0
        word_list.append(w)
        is_keyword.append(label)
        product_name.append(row['Product Name'])
        category.append(row['Category'])
df1 = pd.DataFrame({'words':word_list, 'Category':category,'is_keyword': is_keyword,'Product Name':product_name})[['Product Name','words','Category','is_keyword']]
# 是否在df中有重複的字? : 有,這代表有些字在某個Product Name是關鍵字, 有些不是
print ('Perfect' if df1.shape[0] == df1.words.nunique() else 'ratio that unique words have : {}'.format(1.0 * df1.words.nunique() / df1.shape[0]))
# 有些字在某個Product Name是關鍵字, 有些不是, 那此時要怎麼判斷這個字是不是關鍵字? ps:這裡關鍵字的定義是消費者用來搜尋商品的字
    # 解法1(simple): 用voting的方式,這個word在所有Product Name中是keyword的比例, 但此法假設每個字在不同的Product Name所佔的權重是一樣的。
    # 解法2(make sense): 加入一欄位,描述此word在此Product Name所佔的權重or與此Product Name的相似性
    # 除了Product Name ID:基本上找不到unique value來代表某個Product Name中的某個字

df2 = df1.groupby(['words']).mean().rename( columns = {'is_keyword': 'ratio_of_keywords'}) \
.sort_values(by = 'ratio_of_keywords', ascending = False).reset_index()

col = ['Product Name','words','Category','ratio_of_keywords']

train = pd.merge(df1,df2,on = 'words', how = 'left')[col]
del df1, df2

#-------------------------
# test
#-------------------------

# covert word to lower()
test['Product Name'] = [p_n.lower() for p_n in test['Product Name']]
# okenization
test['product_name_tokens'] = test['Product Name'].apply(extracted_keywords_by_jeiba)
# remove stopwords
test['product_name_tokens'] = test['product_name_tokens'].apply(remove_stopwords)

#--------------
# 針對特殊pattern such as brand name 做處理, for making sure get a better tokenizing result.
#--------------
for ix, row in test[['Product Name','product_name_tokens']].iterrows():
   # 手動處理特殊pattern的分詞
    for s_p in pattern:
        # text preprocessing for Product Name
        if s_p in row['Product Name']:
            # replace query_tokesn with new_tokens if special pattern in query
            test.product_name_tokens.ix[ix] = get_new_token(row['product_name_tokens'], row['product_name_tokens'] + re.findall(s_p, row['Product Name']))

word_list = []
product_name = []
category = []
for ix, row in test[['Product Name','Category','product_name_tokens']].iterrows():    
    for p in row['product_name_tokens']:
        word_list.append(p)
        product_name.append(row['Product Name'])
        category.append(row['Category'])
        
test = pd.DataFrame({'words':word_list,'Product Name':product_name, 'Category': category}) \
[['Product Name','Category','words']]


#-------------------------
#save
#-------------------------

train.to_csv('../input/train_mobile.csv.gz', index = False, compression='gzip')
test.to_csv('../input/test_mobile.csv.gz', index = False, compression='gzip')

