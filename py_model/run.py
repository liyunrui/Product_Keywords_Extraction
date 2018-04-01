#!/usr/bin/python3
# -*-coding:utf-8
'''

@author: Ray

concating features
'''
import pandas as pd
import numpy as np
import os
import utils # written by author

##############
#Modeling
##############

os.system('python3 -u xgb_keyword_extraction.py')

##############
#submit results
##############

os.system('python3 -u keywords_extraction.py')

print ('done')