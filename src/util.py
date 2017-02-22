'''
Created on Jan 1, 2017

@author: anmiko
'''


import pandas as pd
import numpy as np 
from scipy import stats
from scipy import misc

from matplotlib import pyplot as plt 

import os, json, glob
import logging
from time import time,ctime

from keras.models import Sequential
#from keras.layers.core import Flatten, Dense, Dropout, Input
from keras.layers import Input, Dense
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

from keras.applications import vgg16
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import Model
from PIL import Image

from functools import partial
from keras.utils import np_utils

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import SGD

from keras.layers import Input
from keras.layers.pooling import AveragePooling2D

from keras.layers.advanced_activations import PReLU

from keras.applications import vgg16
from keras.applications.vgg16 import VGG16
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import Model

from keras.optimizers import Adam

from numpy.random import random, permutation
from keras.preprocessing import image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import sklearn.metrics as metrics


#import cv2

PROJECT_NAME = 'cat_dog'
PROJECT_DIR = "/" + PROJECT_NAME +"/"

def full_path(fname, dir_ = 'data'):
    DATA_DIR = '../../' + dir_ + PROJECT_DIR
    return DATA_DIR + fname 

#img = Image.open("AFLAC.jpg").transpose(Image.FLIP_LEFT_RIGHT)
def preprocess_image(image_path, target_size = (224,224), flip = None):
    img = load_img(image_path, target_size = target_size)
    if flip is not None:
        img = img.transpose(flip)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img

#from vgg_utils
def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
    if type(ims[0]) is np.ndarray:
        ims = np.array(ims).astype(np.uint8)
        if (ims.shape[-1] != 3):
            ims = ims.transpose((0,2,3,1))
    f = plt.figure(figsize=figsize)
    for i in range(len(ims)):
        sp = f.add_subplot(rows, len(ims)//rows, i+1)
        if titles is not None:
            sp.set_title(titles[i], fontsize=18)
        plt.imshow(ims[i], interpolation=None if interp else 'none')
        
#img = Image.open("AFLAC.jpg").transpose(Image.FLIP_LEFT_RIGHT)
# def prep_x_y(img_df, idx, target_size = (224,224), flip = None):
#     tmp = [preprocess_image(img_df.path[i], target_size = target_size, flip = flip) for i in idx]
#     x = np.concatenate(tmp, axis = 0)
#     y = np.array(img_df.dog[idx]).astype(np.float32)
#     return x,y
# 
# def prep_x(img_df, idx, target_size = (224,224), flip = None):
#     tmp = [preprocess_image(img_df.path[i], target_size = target_size, flip = flip) for i in idx]
#     x = np.concatenate(tmp, axis = 0)
#     return x

def prep_x_y(img_df, target_size = (224,224), flip = None):
    tmp = [preprocess_image(r.path, target_size = target_size, flip = flip) for r in img_df.itertuples(index=False)]
    x = np.concatenate(tmp, axis = 0)
    if 'dog' in img_df:
        y = np.array(img_df.dog).astype(np.float32)
        return x,y
    else:
        return x

def attach_top_to_vgg(m_fun, layer_name = 'block5_conv3'):
    vgg_model = VGG16(weights='imagenet', include_top=False)
    
    for layer in vgg_model.layers:
        layer.trainable = False
    x = vgg_model.get_layer(layer_name).output
    model = Model(vgg_model.input, m_fun(x))
    return model


###generators
def gen_batch_wrap(gen, batch_size = 32):
    while True:
        batch_x = []
        batch_y = []
        for i in range(batch_size):
            x = next(gen)
            if type(x) is tuple:
                batch_y.append(x[1])
                x = x[0]
            batch_x.append(x)
        x_arr = np.concatenate(batch_x, axis = 0)
        if len(batch_y) > 0:
            y_arr = np.concatenate(batch_y, axis = 0)
            yield (x_arr, y_arr)
        else:
            yield x_arr
            
#crop fun
def make_crop_fun(crop_size, max_size = 7):
    logging.info("make_crop_fun called")
    def crop_pool(x):
        #logging.basicConfig(format='%(asctime)s %(message)s', filename=full_path('general.log', 'log'), level=logging.INFO)
        stride = [0,0]
        stride[0] = np.random.randint(max_size - crop_size + 1)
        stride[1] = np.random.randint(max_size - crop_size + 1)
        logging.info("sride: %s", stride)
        res = x[:,stride[0]:crop_size+stride[0],stride[1]:crop_size+stride[1],:]
        #and random flip
        if np.random.binomial(1,0.5):
            #res = np.fliplr(res)
            res = res[:,::-1]
        return res
    return crop_pool

def conv_crop_gen(X, y, crop_fun):
    while True:
        for i in np.random.permutation(X.shape[0]):
            res = (crop_fun(X[i:i+1,::]), y[i:i+1])
            yield res
            
def reduce_conf(pr, high =0.99,low = 0.01):
    y_r = pr.copy()
    y_r[y_r > high] = high
    y_r[y_r < low] = low
    return y_r