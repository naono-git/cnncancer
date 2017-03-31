#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('train stacked autoencoder stage 1')

import os
import sys
import csv
import numpy as np
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
# load sample data
#
dir_data = 'dat1'

nx = 32 # sample size
file_input = 'pancreas_w{}.npy'.format(nx)
path_data = os.path.join(dir_data,file_input)
print('load input from {}'.format(path_data))
qqq_trn = np.load(path_data)

nn,ny,nx,nl = qqq_trn.shape
print('nn ny nx nl',nn,ny,nx,nl)
exec(open('tensorflow_ae_stage1.py').read())

#
# setup optimizer
#
tf_input = tf.placeholder(tf.float32, [None,ny,nx,nl])
tf_encode1 = get_encode1(tf_input)
tf_deconv1 = get_deconv1(tf_encode1)
# local_entropy = get_local_entropy_encode1(tf_encode1)
# mean_entropy = tf.reduce_mean(local_entropy)
# optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
# train = optimizer.minimize(mean_error + lambda_s*mean_entropy)
mean_error = tf.reduce_mean(tf.square(tf_deconv1 - tf_input))
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(mean_error)

sess.run(tf.global_variables_initializer())

#
# train loop
#
iii_bin = np.arange(batch_size,nn,batch_size)
iii_nn = np.arange(nn)
iii_batches = np.split(iii_nn,iii_bin)

for tt in range(tmax):
    if(tt % tprint==0):
        error_out = [sess.run(mean_error,{tf_input: qqq_trn[iii,]}) for iii in iii_batches]
        print(tt,np.mean(error_out))
    np.random.shuffle(iii_nn)
    iii_batches = np.split(iii_nn,iii_bin)
    for iii in iii_batches:
        sess.run(train,feed_dict={tf_input: qqq_trn[iii,]})

tmp = [mean_error.eval({tf_input: qqq_trn[iii,]}) for iii in iii_batches]
error_tmp = np.mean(tmp)
print(tmax,error_tmp)

#
# save parameters
#
save_stage1()

myutil.show_timestamp()
print('stamp1 = \'{}\''.format(stamp))
