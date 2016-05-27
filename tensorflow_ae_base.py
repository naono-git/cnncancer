# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>
print("tensorflow_ae_base")
# <codecell>

import numpy as np
import tensorflow as tf
import pickle

# Create model
def conv2d_no_bias(xxx, ww):
    return(tf.nn.conv2d(xxx, ww, strides=[1, 1, 1, 1], padding='SAME'))

def conv2d(xxx, ww, bb):
    return tf.nn.bias_add(tf.nn.conv2d(xxx, ww, strides=[1, 1, 1, 1], padding='SAME'),bb)

def max_pool(xxx, kk):
    return tf.nn.max_pool(xxx, ksize=[1,kk,kk,1], strides=[1,kk,kk,1], padding='SAME')

def un_pool_2(xxx):
    ddd = xxx.get_shape().dims
    d1 = ddd[1].value
    d2 = ddd[2].value
    d3 = ddd[3].value
    yyy0 = tf.pack([xxx,xxx])
    zzz0 = tf.transpose(yyy0,[1,2,0,3,4])
    xxx1 = tf.reshape(zzz0,[-1,d1*2,d2,d3])
    yyy1 = tf.pack([xxx1,xxx1])
    zzz1 = tf.transpose(yyy1,[1,2,3,0,4])
    xxx2 = tf.reshape(zzz1,[-1,d1*2,d2*2,d3])
    return xxx2

def un_pool(xxx, kk):
    if kk==2:
        xxx2 = un_pool_2(xxx)
        return(xxx2)
    
    ddd = xxx.get_shape().dims
    d1 = ddd[1].value
    d2 = ddd[2].value
    d3 = ddd[3].value
    yyy0 = xxx
    for ii in range(0,kk-1):
        yyy0 = tf.pack([yyy0,xxx])
    zzz0 = tf.transpose(yyy0,[1,2,0,3,4])
    xxx1 = tf.reshape(zzz0,[-1,d1*kk,d2,d3])
    yyy1 = xxx1
    for ii in range(0,kk-1):
        yyy1 = tf.pack([yyy1,xxx1])
    zzz1 = tf.transpose(yyy1,[1,2,3,0,4])
    xxx2 = tf.reshape(zzz1,[-1,d1*kk,d2*kk,d3])
    return xxx2

def load_tf_variable(path_src,keys=None,trainable=True):
    dict_tmp = pickle.load(open(path_src,'rb'))
    ddd = {kk:tf.Variable(dict_tmp[kk],trainable=trainable) for kk in keys}
    for kk in keys:
        print(kk,dict_tmp[kk].shape)
    return(ddd)
