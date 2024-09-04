#!/bin/bash 

"""
author: jdieg@ 
"""


# import libs
from sklearn.model_selection import train_test_split
import os

# categories 
class_list = ['B_lymphoma' , 'T_lymphoma']


# get test and train class 

test_train={}

TILES_TRAIN_PATH ='/media/jdiego/0fcbc4f2-ab1a-4855-894c-f5ca933658e2/jdiego/inter/train/combined_tiles/'    # repalce with your training set path

for class_ in class_list:
    files = os.listdir(TILES_TRAIN_PATH)
    train, test = train_test_split(files, test_size=0.2, random_state=42)
    test_train['{}_train'.format(class_)]= train
    test_train['{}_test'.format(class_)]= test


print(test_train.keys())


TILES_TRAIN_PATH ='/media/jdiego/0fcbc4f2-ab1a-4855-894c-f5ca933658e2/jdiego/inter/train/combined_tiles/'    # repalce with your training set path
TILES_TEST_PATH = '/media/jdiego/0fcbc4f2-ab1a-4855-894c-f5ca933658e2/jdiego/inter/test_tiles/'              # repalce with your test path

for class_ in class_list:
    for file_ in test_train['{}_test'.format(class_)]:
        os.rename(os.path.join(TILES_TRAIN_PATH,file_), os.path.join(TILES_TEST_PATH, class_, file_))



validate_train={}
TILES_TRAIN_PATH ='/media/jdiego/0fcbc4f2-ab1a-4855-894c-f5ca933658e2/jdiego/inter/train/'                   # repalce with your training set path
for class_ in class_list:
    files = os.listdir(os.path.join(TILES_TRAIN_PATH, class_))
    train, validate = train_test_split(files, test_size=0.2, random_state=42)
    validate_train['{}_train'.format(class_)]= train
    validate_train['{}_test'.format(class_)]= validate


# function to store in PATHS 
TILES_TRAIN_PATH ='/media/jdiego/0fcbc4f2-ab1a-4855-894c-f5ca933658e2/jdiego/inter/train'                            # repalce with your training set path
TILES_VALIDATION_PATH = '/media/jdiego/0fcbc4f2-ab1a-4855-894c-f5ca933658e2/jdiego/inter/validation_tiles'           # repalce with your validation path
for class_ in class_list:
    for file_ in validate_train['{}_test'.format(class_)]:
        os.rename('{}/{}/{}'.format(TILES_TRAIN_PATH, class_,file_), '{}/{}/{}'.format(TILES_VALIDATION_PATH, class_, file_))
