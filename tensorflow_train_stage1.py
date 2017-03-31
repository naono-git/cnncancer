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

ss = 32 # sample size
if(not 'qqq_trn' in locals()):
    path_data = os.path.join(dir_data,'tcga_trn_w32.npy')
    qqq_trn = np.load(path_data)
    print('load input from {}'.format(path_data))

nn,ny,nx,nl = qqq_trn.shape
print('nn ny nx nl',nn,ny,nx,nl)
exec(open('tensorflow_ae_stage1.py').read())

#
# setup optimizer
#
qqq_input = tf.placeholder(tf.float32, [None,ny,nx,nl])
qqq_encode1 = get_encode1(qqq_input)
qqq_deconv1 = get_deconv1(qqq_encode1)
mean_error = tf.reduce_mean(tf.square(qqq_deconv1 - qqq_input))
#local_entropy = get_local_entropy_encode1(qqq_encode1)
#mean_entropy = tf.reduce_mean(local_entropy)
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(mean_error + lambda_s*mean_entropy)
## train = optimizer.minimize(mean_error)

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
        error_out = np.mean([sess.run(mean_error,{qqq_input: qqq_trn[iii,]}) for iii in iii_batches])
        print(tt,error_out)
    np.random.shuffle(iii_nn)
    iii_batches = np.split(iii_nn,iii_bin)
    for iii in iii_batches:
        sess.run(train,feed_dict={qqq_input: qqq_trn[iii,]})

#
# save parameters
#
save_stage1()
pickle.dump(network_params,open('network_params.{}.pkl'.format(stamp),'wb'))

myutil.timestamp()
print('stamp1 = \'{}\''.format(stamp))
