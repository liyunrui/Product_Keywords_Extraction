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
# Female_Fashion
###########################
train = sample[sample.Category == 'Female Fastion']
train['Category'] = train['Category'].map(lambda x: 'Female Clothes') # making this field of train set consistent with test set
test = test[test.Category == 'Female Clothes']
test = test.reset_index().drop('index', axis = 1)
def extracted_keywords_by_jeiba(x):
    '''
    Convert Product name to tokens
    parameters:
    --------------
    x: str
    '''
    # Normalizing相同意義的字(in order to reudce vector space of words)
    x = x.replace('短t','t恤')
    x = x.replace('t shirt','t恤')
    x = x.replace('alasha', 'a la sha')
    x = x.replace('brandy', 'brandy melville')
    x = x.replace('愛迪達', 'adidas') 
    # 添加自定义词
    for w in ['女生衣著','韓版','t恤','你別睡','七分褲','五分褲','長裙','前扣式','棒球外套','marvel',
    '優衣庫','森林系','細圖','牛仔','大尺碼','波希米亞','長洋裝','小可愛','長版','歐美','雪紡',
    '縮口褲','連身褲','洋裝','假兩件','蛋糕裙','大學t','娃娃裝','綁帶','格子','顯瘦','夏天',
    '細肩帶','洋裝','兩件式','洋裝','正韓','刺繡','上衣','休閒','學院風','蕾絲','米奇','短褲',
    '削肩','民族風','爆汗褲','男友褲','吊帶褲','短袖','polo衫','寬褲','丁字褲','哈倫褲','個性',
    '針織','背心','百褶裙','莫代爾','火焰','肩帶','復古','排汗衫','排汗','背心','荷葉邊','露肩',
    '西裝褲','襯衫','防曬','迷彩','長褲','刺繡','免運','現貨','仙人掌','直筒褲','東京著衣',
    '無袖','維多利亞的秘密','情侶','海灘褲','文字t','潮牌','上衣',
    ]:
        jieba.add_word(w)
    # 加载自定义词库
    #jieba.load_userdict("../input/customized_dict_for_Mobile_Gadgets.txt")
    tokens = None
    if len(jieba.analyse.extract_tags(x, topK = len(x))) != 0:
        tokens = jieba.analyse.extract_tags(x, topK = len(x))
    else:
        tokens = [x]
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

# covert word to smaller
train['Query'] = [q.lower() for q in train['Query']]
train['Product Name'] = [p_n.lower() for p_n in train['Product Name']]

# Tokenization
train['product_name_tokens'] = train['Product Name'].apply(extracted_keywords_by_jeiba)
train['query_tokens'] = train['Query'].apply(extracted_keywords_by_jeiba)
# create stopwords, which are defined by the words that are not exactly used to search prodcut by users.
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
pattern = ['hollister','scottish house','adidas originals','pazzo','ua','gap','marjorie','h&m','zara',
           'voda swim','mercci22','roxy','champion','adidas','t shirt','a la sha','air space','roots',
           'anden hud','brandy melville','h connect','hush puppies','queen shop','uniqlo','richmond&finch',
           'tommy hilfiger','carhartt','oversize','interbreed','prettynice','fila','stussy',
           'timberland','dickies','lolita','bikini','eyescream',
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

train.to_csv('../input/train_female_fashion.csv.gz', index = False, compression='gzip')
test.to_csv('../input/test_female_fashion.csv.gz', index = False, compression='gzip')

