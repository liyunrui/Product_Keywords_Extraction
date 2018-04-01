#!/usr/bin/python3
# -*-coding:utf-8
'''
Created on Sun Mar 18 2018

@author: Ray

'''

import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from xgboost import plot_importance
from xgboost import XGBRegressor
import time
from datetime import datetime
from xgboost import plot_importance
import numpy as np
import pickle # for saving
import utils # made by author for efficiently dealing with data
import gc
from sklearn.model_selection import train_test_split
###############################
# setting
###############################

# using different random seed to make sure variety of models
seed = 1992
np.random.seed(seed) 

DATE = '0328'
utils.mkdir_p('../output/model/{}_{}/'.format(DATE,seed))
utils.mkdir_p('../output/sub/{}_{}/'.format(DATE,seed))

print("""#==== print param ======""")
print('DATE:', DATE)
print('seed:', seed)

##################################
# loading data
##################################
train = pd.read_csv('../feature/{}/all_features.csv.gz'.format('train'), compression='gzip')

#==============================================================================
# prepare training data
#==============================================================================
Y_train = train['ratio_of_keywords'] 
X_train = train.drop(['Product Name','words','ratio_of_keywords'], axis=1)

print ('prepartion of training set is done')


#==============================================================================
print('training')
#==============================================================================

# setting

default_params = {
    'objective': 'reg:linear',
    'max_depth': 6,
    'min_child_weight': 1,
    'gamma' : 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'silent': 1.0
}
early_stopping_rounds = 50
learning_rate = 0.01
n_estimators = 20000

# for simple ensemble
LOOP = 2
# Core
models = [] # for the following prediction
for i in range(LOOP):
    print('LOOP',i)
    # hold-out validation
    x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state= seed)

    # model training
    s = time.time()
    model = XGBRegressor(
        **default_params,
        n_estimators = n_estimators,
        learning_rate = learning_rate,
        seed = seed 
                             )
    model.fit(x_train, y_train, 
          eval_metric ='rmse' ,eval_set = [(x_val, y_val)],
          early_stopping_rounds = early_stopping_rounds) 
    e = time.time()
    print ('sec : ', e - s) # 17065.98352408409 sec=~ 4.74 hours
    models.append(model)
    #save model
    pickle.dump(model, open('../output/model/{}_{}/xgb_keywords_extraction_{}.model'.format(DATE, seed, i), "wb"))
    # validating
    valid_yhat = model.predict(x_val) # y_hat is result of prediction
    print('Valid Mean:', np.mean(valid_yhat))

del X_train, Y_train

#==============================================================================
print('test')
#==============================================================================
test = pd.read_csv('../feature/{}/all_features.csv.gz'.format('test'), compression='gzip')

sub_test = test[['Product Name', 'words']]
sub_test['yhat'] = 0
col = x_train.columns.tolist()

for model in models:
    sub_test['yhat'] += model.predict(test[col])
sub_test['yhat'] /= LOOP # do some simple ensemble: average of prediting result    

print('Test Mean:', sub_test['yhat'].mean())

'''
how to load xgboost mode with pickle
models = [
    pickle.load(open("../output/model/0328_1992/xgb_churn_0.model", "rb")),
    pickle.load(open("../output/model/0328_1992/xgb_churn_1.model", "rb"))
]
'''


# Convert yhat into prob given Product Name
def yhat_in_range_0_to_1(x):
    '''
    To obtain probability that belongs to keyword given the product name by min-max tranformation
    
    parameters:
    --------------
    x:DataFrame
    '''
    if x.shape[0] == 1:
        x['yhat_in_range_0_to_1'] = x.yhat
    else:
        x['yhat_in_range_0_to_1'] = (x.yhat - x.yhat.min())/ (x.yhat.max() - x.yhat.min())
    return x
def convert_prob(x):
    '''
    To obtain probability that belongs to keyword given the product name by min-max tranformation
    
    parameters:
    --------------
    x:DataFrame
    '''
    x['prob_given_the_product_name'] = (x.yhat_in_range_0_to_1)/ (x.yhat_in_range_0_to_1.sum())
    return x

sub_test = sub_test.groupby(['Product Name']).apply(yhat_in_range_0_to_1)
sub_test = sub_test.groupby(['Product Name']).apply(convert_prob)

print ('opps' if sub_test.isnull().values.any() else 'No NaN value in sub_test')
# saving for picking up the best keyword
output = ['Product Name','words','prob_given_the_product_name']
sub_test[output].to_excel('../output/sub/{}_{}/sub_test.xlsx'.format(DATE,seed), index = False) # to_excel: for chinese
