#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('train stacked autoencoder stage 1')

import os
import sys
import csv
import numpy as np
import pickle
from PIL import Image

import tensorflow as tf

import tensorflow_ae_base
from tensorflow_ae_base import *
import tensorflow_util
import myutil

exec(open('extern_params.py').read())

#
# load input data
#
ss = 32 # sample size
if(not 'qqq_trn' in locals()):
    file_input = 'qqq_trn_w{}.npy'.format(ss)
    path_data = os.path.join(dir_input,'input_w{}'.format(ss),file_input)
    qqq_trn = np.load(path_data)
    print('load input from {}'.format(path_data))

nn,ny,nx,nl = qqq_trn.shape
print('nn ny nx nl',nn,ny,nx,nl)

nf_RGB = 3
nf_conv1 = 12

ww = tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_RGB,12],stddev=0.05))
tf_input = tf.placeholder(tf.float32, [None,ny,nx,nl])

tf_conv1 = tf.nn.conv2d(tf_input,ww,strides=[1,1,1,1],padding='SAME')
tf_deconv1 = tf.nn.conv2d_transpose(value=tf_conv1,filter=ww,
                                    output_shape=[batch_size,ny,nx,nl],
                                    strides=[1,1,1,1],
                                    padding='SAME')
## tf_error = tf.reduce_mean(tf.square(tf_deconv1 - tf_input))
tf_error = tf.nn.l2_loss(tf_deconv1 - tf_input)

tf_simple1 = tf.square(tf_conv1)
seg12 = tf.constant([1,1,2,2,3,3,4,4,5,5,6,6])
tf_t_simple1 = tf.transpose(tf_simple1)
tf_sparce1 = tf.reduce_mean(tf.sqrt(tf.segment_sum(tf_t_simple1,vv)))

tf_cost = tf_error + lambda_s * tf_sparce1
# tf_cost = tf_error + lambda_s * tf_sparce + lambda_n * tf_norm

optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(tf_cost)

sess.run(tf.initialize_all_variables())

iii_bin = np.arange(batch_size,nn,batch_size)
iii_nn = np.arange(nn)
iii_batches = np.split(iii_nn,iii_bin)

for tt in range(tmax):
    if(tt % tprint==0):
        tmp = [sess.run(mean_error,{tf_input: qqq_trn[iii,]}) for iii in iii_batches]
        error_out = np.mean(tmp)
        print(tt,error_out)
    np.random.shuffle(iii_nn)
    iii_batches = np.split(iii_nn,iii_bin)
    for iii in iii_batches:
        sess.run(train,feed_dict={tf_input: qqq_trn[iii,]})
