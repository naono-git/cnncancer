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

# extern stamp1c
exec(open('extern_params.py').read())

#
# load sample data
#

ss = 32 # sample size

file_input = 'pancreas_encode1_w{}.{}.npy'.format(ss,stamp1)
path_data = os.path.join(dir_data,file_input)
print('load input from {}'.format(path_data))
qqq_encode1 = np.load(path_data)

nn,ny,nx,nl = qqq_encode1.shape
print('nn ny nx nl',nn,ny,nx,nl)

# nf_encode1
exec(open('tensorflow_ae_stage2.py').read())

#
# setup optimizer
#
tf_input = tf.placeholder(tf.float32, [None,ny,nx,nf_encode1])
tf_encode2 = get_encode2(tf_input)
tf_deconv2 = get_deconv2(tf_encode2)
mean_error = tf.reduce_mean(tf.square(tf_deconv2 - tf_input))
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(mean_error)

sess.run(tf.global_variables_initializer())

#
# train loop
#
iii_bin = np.arange(batch_size,nn,batch_size)
iii_nn = np.arange(nn)
iii_batches = np.split(iii_nn,iii_bin)

# extern
# tmax,tprint = 10,1
for tt in range(tmax):
    if(tt % tprint==0):
        tmp = [mean_error.eval({tf_input: qqq_encode1[iii,]}) for iii in iii_batches]
        error_tmp = np.mean(tmp)
        print(tt,error_tmp)
    np.random.shuffle(iii_nn)
    iii_batches = np.split(iii_nn,iii_bin)
    for iii in iii_batches:
        sess.run(train,feed_dict={tf_input: qqq_encode1[iii,]})

if(tt < tmax):
    tmp = [mean_error.eval({tf_input: qqq_encode1[iii,]}) for iii in iii_batches]
    error_tmp = np.mean(tmp)
    print(tmax,error_tmp)

#
# save parameters
#
save_stage2()

# exec(open('tensorflow_verify_stage2.py').read())
# exec(open('tensorflow_encode_stage2.py').read())
myutil.show_timestamp()
print('stamp2 = \'{}\''.format(stamp))
