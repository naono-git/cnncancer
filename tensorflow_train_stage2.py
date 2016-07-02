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
exec(open('tensorflow_ae_stage2.py').read())

#
# setup optimizer
#
tf_input = tf.placeholder(tf.float32, [None,ny,nx,nl])
tf_encode1 = get_encode1(tf_input)
tf_encode2 = get_encode2(tf_encode1)
tf_deconv2 = get_deconv2(tf_encode2)
tf_deconv1 = get_deconv1(tf_deconv2)
mean_error = tf.reduce_mean(tf.square(tf_deconv1 - tf_input))
local_entropy = get_local_entropy_encode2(tf_encode2)
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
        error_out = np.mean([sess.run(mean_error,{tf_input: qqq_trn[iii,]}) for iii in iii_batches])
        print(tt,error_out)
    np.random.shuffle(iii_nn)
    iii_batches = np.split(iii_nn,iii_bin)
    for iii in iii_batches:
        sess.run(train,feed_dict={tf_input: qqq_trn[iii,]})

#
# save parameters
#
save_stage1()
save_stage2()

myutil.timestamp()
print('stamp2 = \'{}\''.format(stamp))

