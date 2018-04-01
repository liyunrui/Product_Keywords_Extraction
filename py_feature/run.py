#!/usr/bin/python3
# -*-coding:utf-8
'''

@author: Ray

concating features
'''
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import utils # written by author
import gc # for automatic releasing memory

##############
#Text prepprocessing
##############

os.system('python3 -u text_preprocessing_mobile.py')
os.system('python3 -u text_preprocessing_female.py')
os.system('python3 -u text_preprocessing_male.py')

print ('Processing done')
##############
#Feature Engineering
##############

os.system('python3 -u relationship_between_word_and_productname.py')
os.system('python3 -u is_the_words_special_brand.py')
os.system('python3 -u hot_search_feature.py')
os.system('python3 -u semantic_feature.py')

print ('Feature Engineering done')

##############
#concat
##############

os.system('python3 -u concat.py')
print ('concat done')