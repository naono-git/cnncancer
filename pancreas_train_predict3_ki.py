#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('train stage2 predictor of ki57')

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
from myutil import source

exec(open('extern_params.py').read())

#
# load sample data
#
dir_data = 'dat1'
if(not ('qqq_src' in locals() or 'qqq_dst' in locals())):
    file_src = 'datafile_w1024_he_1.npy'
    path_src = os.path.join(dir_data,file_src)
    data_src = np.load(path_src)
    print('load input from {}'.format(path_src))

    file_dst = 'datafile_w1024_ki_1.npy'
    path_dst = os.path.join(dir_data,file_dst)
    data_dst = np.load(path_dst)
    print('load input from {}'.format(path_dst))

np.random.seed(random_seed)

ns = 10000
qqq_src = np.zeros((ns,32,32,3))
qqq_dst = np.zeros((ns,32,32,3))
for aa in range(ns):
    i0 = np.random.choice(range(1024-32-100),size=1)[0]+100
    j0 = np.random.choice(range(1024-32-100),size=1)[0]+100
    i1 = i0+32
    j1 = j0+32
    qqq_src[aa,:]=data_src[j0:j1,i0:i1,:]
    qqq_dst[aa,:]=data_dst[j0:j1,i0:i1,:]

nn,ny,nx,nl = qqq_src.shape
print('nn ny nx nl',nn,ny,nx,nl)
trainable2=True
trainable1=False
exec(open('tensorflow_ae_stage1.py').read())
exec(open('pancreas_ki2.py').read())

#
# setup optimizer
#
tf_src = tf.placeholder(tf.float32, [None,ny,nx,nl])
tf_dst = tf.placeholder(tf.float32, [None,ny,nx,nl])
tf_dst2 = get_dist2(tf_dst)
tf_encode1 = get_encode1(tf_src)
tf_predict = get_predict2(tf_encode1)
mean_error = tf.reduce_mean(tf.square(tf_predict - tf_dst2))
optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(mean_error)

#
# train loop
#
iii_bin = np.arange(batch_size,nn,batch_size)
iii_nn = np.arange(nn)
iii_batches = np.split(iii_nn,iii_bin)

sess.run(tf.initialize_all_variables())

for tt in range(tmax):
    if(tt % tprint==0):
        error_out = np.mean([sess.run(mean_error,{tf_src: qqq_src[iii,],tf_dst:qqq_dst[iii,]}) for iii in iii_batches])
        print(tt,error_out)
    np.random.shuffle(iii_nn)
    iii_batches = np.split(iii_nn,iii_bin)
    for iii in iii_batches:
        sess.run(train,feed_dict={tf_src: qqq_src[iii,],tf_dst:qqq_dst[iii,]})

#
# save parameters
#
save_stage1()
save_stage2()
myutil.timestamp()
print('stamp2 = \'{}\''.format(stamp))
