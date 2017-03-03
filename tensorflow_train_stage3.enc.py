#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('train stacked autoencoder stage 3')

import os
import sys
import csv
import numpy as np
import pickle
from PIL import Image

import tensorflow as tf

import tensorflow_ae_base
from   tensorflow_ae_base import *
import tensorflow_util
import myutil

# extern stamp2
exec(open('extern_params.py').read())

#
# load sample data
#

ss = 32 # sample size # w64 encoded to 32

file_input = 'tcga_encode2_w{}.{}.npy'.format(ss,stamp2)
path_data = os.path.join(dir_data,file_input)
print('load input from {}'.format(path_data))
qqq_encode2 = np.load(path_data)

nn,ny,nx,nl = qqq_encode2.shape
print('nn ny nx nl',nn,ny,nx,nl)

# extern stamp3
exec(open('tensorflow_ae_stage3.py').read())

#
# setup optimizer
#
tf_input = tf.placeholder(tf.float32, [None,ny,nx,nl])
tf_encode3 = get_encode3(tf_input)
tf_deconv3 = get_deconv3(tf_encode3)
mean_error = tf.reduce_mean(tf.square(tf_deconv3 - tf_input))
local_entropy = get_local_entropy_encode3(tf_encode3)
mean_entropy = tf.reduce_mean(local_entropy)
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(mean_error + lambda_s*mean_entropy)

sess.run(tf.initialize_all_variables())

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
        tmp = [sess.run([mean_error,mean_entropy],{tf_input: qqq_encode2[iii,]}) for iii in iii_batches]
        error_out = np.mean([xx[0] for xx in tmp])
        entropy_out = np.mean([xx[1] for xx in tmp])
        print(tt,error_out,entropy_out, error_out+lambda_s*entropy_out)
    np.random.shuffle(iii_nn)
    iii_batches = np.split(iii_nn,iii_bin)
    for iii in iii_batches:
        sess.run(train,feed_dict={tf_input: qqq_encode2[iii,]})

if(tt < tmax):
    tmp = [sess.run([mean_error,mean_entropy],{tf_input: qqq_encode2[iii,]}) for iii in iii_batches]
    error_out = np.mean([xx[0] for xx in tmp])
    entropy_out = np.mean([xx[1] for xx in tmp])
    print(tmax,error_out,entropy_out, error_out+lambda_s*entropy_out)

#
# save parameters
#
save_stage3()

myutil.timestamp()
print('stamp3 = \'{}\''.format(stamp))
