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
na = 4

if((not 'qqq_encode1' in locals()) or (not qqq_encode1.shape[2] == ss)):
    tmp = []
    for aa in range(na):
        file_input = 'qqq_encode1_tcga_w{}_{}.{}.npy'.format(ss,aa+1,stamp1)
        path_data = os.path.join(dir_out,file_input)
        tmp.append(np.load(path_data))
        print('load input from {}'.format(path_data))
    qqq_encode1 = np.vstack(tmp)

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

# extern
# tmax,tprint = 10,1
for tt in range(tmax):
    if(tt % tprint==0):
        tmp = [sess.run([mean_error,mean_entropy],{tf_input: qqq_encode1[iii,]}) for iii in iii_batches]
        error_out = np.mean([xx[0] for xx in tmp])
        entropy_out = np.mean([xx[1] for xx in tmp])
        print(tt,error_out,entropy_out, error_out+lambda_s*entropy_out)
    np.random.shuffle(iii_nn)
    iii_batches = np.split(iii_nn,iii_bin)
    for iii in iii_batches:
        sess.run(train,feed_dict={tf_input: qqq_encode1[iii,]})

if(tt < tmax):
    tmp = [sess.run([mean_error,mean_entropy],{tf_input: qqq_encode1[iii,]}) for iii in iii_batches]
    error_out = np.mean([xx[0] for xx in tmp])
    entropy_out = np.mean([xx[1] for xx in tmp])
    print(tmax,error_out,entropy_out, error_out+lambda_s*entropy_out)

#
# save parameters
#
weight2_fin = {k:sess.run(v) for k,v in weight2.items()}
bias2_fin = {k:sess.run(v) for k,v, in bias2.items()}
myutil.saveObject(weight2_fin,'weight2.{}.pkl'.format(stamp))
myutil.saveObject(bias2_fin,'bias2.{}.pkl'.format(stamp))

# exec(open('tensorflow_verify_stage2.py').read())
# exec(open('tensorflow_encode_stage2.py').read())
myutil.timestamp()
print('stamp2 = \'{}\''.format(stamp))
