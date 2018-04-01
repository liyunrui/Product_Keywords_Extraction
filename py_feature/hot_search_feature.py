import pandas as pd
import numpy as np
import jieba
import jieba.analyse
import re
import warnings
import itertools

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


df0 = pd.read_csv('../input/train_mobile.csv.gz', compression='gzip')
df1 = pd.read_csv('../input/train_female_fashion.csv.gz', compression='gzip')
df2 = pd.read_csv('../input/train_male_fashion.csv.gz', compression='gzip')
train = pd.concat([df0, df1, df2], ignore_index=True)

df0 = pd.read_csv('../input/test_mobile.csv.gz', compression='gzip')
df1 = pd.read_csv('../input/test_female_fashion.csv.gz', compression='gzip')
df2 = pd.read_csv('../input/test_male_fashion.csv.gz', compression='gzip')
test = pd.concat([df0, df1, df2], ignore_index=True)


def extracted_keywords_by_jeiba(x):
    '''
    Convert Product name to tokens
    parameters:
    --------------
    x: str
    '''
    # Normalizing相同意義的字
    x = x.replace('+',' plus')
    x = x.replace('i phone','iphone')
     
    
    # 添加自定义词
    for w in [
'無線充電',
 '上衣',
 '踢不爛',
 '華為',
 '個性',
 '小可愛',
 '假兩件',
 '丁字褲',
 '卡娜赫拉',
 '長裙',
 '小熊維尼',
 't恤',
 '爆汗褲',
 '細圖',
 '排汗',
 'sony xa',
 '懶人支架',
 '大理石',
 '七分褲',
 '短袖',
 '前扣式',
 '保護貼',
 '荷葉邊',
 '書本套',
 '無袖',
 '復古',
 '玻璃貼',
 '海灘褲',
 'android保護殼',
 '棒球外套',
 '手機架',
 '東京著衣',
 '手機殼',
 '行動電源',
 '安卓',
 '網卡',
 '民族風',
 '膝上短褲',
 '哈倫褲',
 '米奇',
 '老帽',
 '削肩',
 '襯衫',
 '王者榮耀',
 '直筒褲',
 '學院風',
 '洋裝',
 '展示機',
 '男生衣著',
 '五分袖',
 '森林系',
 '隨身碟',
 '耐及',
 '仙人掌',
 '小飛象',
 '手機支架',
 '空機',
 '螢幕',
 '傳說對決',
 '小米max2',
 '小米 max2',
 '長褲',
 '短褲',
 '寶可夢',
 '紅米note4x',
 '現貨',
 'type c',
 '網路卡',
 '衝浪褲',
 '軟殼',
 'android充電傳輸',
 '汽車手機架',
 '寬褲',
 '吸盤搖桿',
 '露肩',
 '三星',
 '手機套',
 '貓咪大戰爭',
 '灌籃高手',
 '中國風',
 '運動耳機',
 '莫代爾',
 '顯瘦',
 '雪紡',
 '男友褲',
 '背心',
 '長版',
 '免運',
 '快充線',
 '嚕嚕米',
 '百褶裙',
 '迷彩',
 '潮t',
 '吊帶褲',
 '文字t',
 '四角褲',
 '波希米亞',
 '綁帶',
 '紅米note3',
 '刺繡',
 '運動褲',
 '防曬',
 '情侶',
 '防摔',
 '細肩帶',
 '娃娃裝',
 '藍芽耳機',
 '愛迪達',
 '大學t',
 '無臉男',
 '客制化',
 '火烈鳥',
 '休閒',
 '達菲',
 '綁帶洋裝',
 '氣囊支架',
 '自拍棒',
 '充電線',
 '蕾絲',
 'marvel',
 '排汗衫',
 '北極熊',
 '夏天',
 '客製化',
 '女生衣著',
 '維多利亞的秘密',
 '火焰',
 '華碩',
 '西裝褲',
 '廣角鏡頭',
 '優衣庫',
 '蛋糕裙',
 '零件機',
 '長洋裝',
 '大尺碼',
 '你別睡',
 '歐美',
 '美少女戰士',
 '五分褲',
 '格子',
 '連身褲',
 '窄褲',
 '牛仔',
 '縮口褲',
 '太陽能',
 '肩帶',
 '紅米note4',
 '史迪奇',
 '熊熊遇見你',
 '韓版',
 'iphone充電傳輸',
 'iphone保護殼',
 '儲值卡',
 '紅米4x',
 '三星j730',
 '飛天小女警',
 '正韓',
 'polo衫',
 '老人機',
 '沙灘褲',
 '智障手機',
 '兩件式',
 '針織',
 '手機貼紙',
 '防塵塞'
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

#--------------
# preprocessing
#--------------    
sample['Query'] = [q.lower() for q in sample['Query']]
sample['query_tokens'] = sample['Query'].apply(extracted_keywords_by_jeiba)
#--------------
# 針對特殊pattern such as brand name 做處理, for making sure get a better tokenizing result.
#--------------
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
 'iphone6'
  ]
pattern.sort(key = lambda s: len(s)) # sorted by length of str for handling wiht problem: 'oppo r9' and 'oppo r9s'

for ix, row in sample[['Product Name','Query','query_tokens']].iterrows():
    # 手動處理特殊pattern的分詞
    for s_p in pattern:
        # text preprocessing for query
        if s_p in row['Query']:
            # replace query_tokesn with new_tokens if special pattern in query
            sample.query_tokens.ix[ix] = get_new_token(row['query_tokens'], row['query_tokens'] + re.findall(s_p, row['Query']))

query_for_state = list(itertools.chain.from_iterable(sample.query_tokens.tolist()))
hot_search = pd.Series(query_for_state).value_counts().to_frame('count') \
.reset_index().rename(columns = {'index':'hot_search'})
hot_search.to_csv('../input/hot_search.csv', index = False)



hot_search = hot_search.rename(columns = {'hot_search':'words'})
#-------------------------
# train
#-------------------------
train = pd.merge(train, hot_search, on = 'words', how = 'left').fillna(0)[ ['Product Name','words','count']] \
.rename(columns = {'count': 'count_in_hot_search'})

#-------------------------
# test
#-------------------------
test = pd.merge(test, hot_search, on = 'words', how = 'left').fillna(0)[ ['Product Name','words','count']] \
.rename(columns = {'count': 'count_in_hot_search'})


#-------------------------
#save
#-------------------------

train.to_csv('../feature/{}/hot_search_count.csv.gz'.format('train'), index = False, compression='gzip')
test.to_csv('../feature/{}/hot_search_count.csv.gz'.format('test'), index = False, compression='gzip')



