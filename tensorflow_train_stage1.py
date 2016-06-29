#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('train stacked autoencoder stage 1')

import os
import sys
import csv
import pickle
import numpy as np
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

ss = 32 # sample size
if(not 'qqq_trn' in locals()):
    file_input = 'qqq_trn_w{}.npy'.format(ss)
    path_data = os.path.join(dir_input,'input_w{}'.format(ss),file_input)
    qqq_trn = np.load(path_data)
    print('load input from {}'.format(path_data))

nn,ny,nx,nl = qqq_trn.shape
print('nn ny nx nl',nn,ny,nx,nl)
exec(open('tensorflow_ae_stage1.py').read())

#
# setup optimizer
#
qqq_input = tf.placeholder(tf.float32, [None,ny,nx,nl])
qqq_encode1 = get_encode1(qqq_input,weight1)
qqq_deconv1 = get_deconv1(qqq_encode1,weight1)
mean_error = tf.reduce_mean(tf.square(qqq_deconv1 - qqq_input))
local_entropy = get_local_entropy_encode1(qqq_encode1)
mean_entropy = tf.reduce_mean(local_entropy)
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(mean_error + lambda_s*mean_entropy)

#
# train loop
#
iii_bin = np.arange(batch_size,nn,batch_size)
iii_nn = np.arange(nn)
iii_batches = np.split(iii_nn,iii_bin)

sess.run(tf.initialize_all_variables())

for tt in range(tmax):
    if(tt % tprint==0):
        error_out = np.mean([sess.run(mean_error,{qqq_input: qqq_trn[iii,]}) for iii in iii_batches])
        print(tt,error_out)
    np.random.shuffle(iii_nn)
    iii_batches = np.split(iii_nn,iii_bin)
    for iii in iii_batches:
        sess.run(train,feed_dict={qqq_input: qqq_trn[iii,]})

#
# save parameters
#
weight1_fin = {k:sess.run(v) for k,v in weight1.items()}
bias1_fin = {k:sess.run(v) for k,v, in bias1.items()}
myutil.saveObject(weight1_fin,'weight1.{}.pkl'.format(stamp))
myutil.saveObject(bias1_fin,'bias1.{}.pkl'.format(stamp))

myutil.timestamp()
print('stamp1 = \'{}\''.format(stamp))

