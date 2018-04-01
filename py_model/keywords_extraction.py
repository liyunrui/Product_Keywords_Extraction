#!/usr/bin/python3
# -*-coding:utf-8
'''
Created on Sun Mar 18 2018

@author: Ray

'''

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import numpy as np
from operator import itemgetter
from opt_fscore import get_best_prediction
import utils
import multiprocessing as mp
#==============================================================================
# optimize
#==============================================================================
def multi(i):
    '''
    The final list_of_recommended_keywords is chosen by using these probabilities and 
    choosing the words subset with maximum expected F1-score.

    parameters:
    ------------
    i: int
    '''
    words = sub.ix[i]['words']
    preds = sub.ix[i]['prob_given_the_product_name']
    ret = get_best_prediction(words, preds, pNone = 0.0)
    return ret


#------------------>
# read xlsx
#------------------>
seed = 1992
DATE = '0328'
sup_prob = pd.read_excel('../output/sub/{}_{}/sub_test.xlsx'.format(DATE,seed))    

#------------------>
#keywords extraction
#------------------>
# change dataframe for multi
sup_prob = sup_prob.groupby(['Product Name','words']).prob_given_the_product_name.mean().reset_index()
sub = sup_prob.groupby('Product Name').words.apply(list).to_frame()
sub['prob_given_the_product_name'] = sup_prob.groupby('Product Name').prob_given_the_product_name.apply(list)

# start!!!
pool = mp.Pool(4)
callback = pool.map(multi, [i for i in range(sub.shape[0])])
sub['list_of_recommended_keywords'] = callback
sub = sub.reset_index()
#-------------------------
#save
#-------------------------
utils.mkdir_p('../output/sub/final')
output = ['Product Name','list_of_recommended_keywords']
sub[output].to_excel('../output/sub/final/recommended_keywords.xlsx', index = False) # to_excel: for chinese


