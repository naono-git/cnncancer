#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('train stage3 predictor of ki57')

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

ss = 32
#
# load sample data
#
file_src = 'pancreas_he_w{}.npy'.format(ss)
path_src = os.path.join(dir_data,file_src)
print('load input from {}'.format(path_src))
qqq_src = np.load(path_src)

file_dst = 'pancreas_ki_w{}.npy'.format(ss)
path_dst = os.path.join(dir_data,file_dst)
print('load input from {}'.format(path_dst))
qqq_dst = np.load(path_dst)

nn,ny,nx,nl = qqq_src.shape
print('nn ny nx nl',nn,ny,nx,nl)
exec(open('tensorflow_ae_stage1.py').read())
exec(open('tensorflow_ae_stage2.py').read())
exec(open('pancreas_layer_predict3_ki.py').read())

#
# setup optimizer
#
tf_src = tf.placeholder(tf.float32, [None,ny,nx,nl])
tf_dst = tf.placeholder(tf.float32, [None,ny,nx,nl])
tf_dst3 = get_dist3(tf_dst)
tf_encode1 = get_encode1(tf_src)
tf_encode2 = get_encode2(tf_encode1)
tf_predict = get_predict3(tf_encode2)
mean_error = tf.reduce_mean(tf.square(tf_predict - tf_dst3))
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
        error_out = [mean_error.eval({tf_src: qqq_src[iii],tf_dst:qqq_dst[iii]})
                     for iii in iii_batches]
        print(tt,np.mean(error_out))
    np.random.shuffle(iii_nn)
    iii_batches = np.split(iii_nn,iii_bin)
    for iii in iii_batches:
        sess.run(train,feed_dict={tf_src: qqq_src[iii,],tf_dst:qqq_dst[iii,]})

#
# save parameters
#
myutil.show_timestamp()

if(trainable1):
    save_stage1()
if(trainable2):
    save_stage2()
if(True):
    save_stage3()
    print('stamp3 = \'{}\''.format(stamp))
