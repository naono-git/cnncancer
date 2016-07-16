#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('train classyfiler stage 4')


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

ss = 2048 # sample size
na = 1

## one batch one file
# bronchioid
# magnoid
# squamoid	

import random

data_table = list(csv.reader(open('filelist.txt','r'), delimiter='\t'))
ns = len(data_table)


ni = 16
iii = random.sample(range(ns),ni)
tmp = []
tmpy = []
for aa in range(ni):
    ii = iii[aa]
    path_data = data_table[ii][0]
    img_tmp = Image.open(path_data,'r')
    tmp.append((np.asarray(img_tmp) / 255.0)[np.newaxis,:,:,:])
    tmpy.append(int(data_table[ii][1]))
qqq_trn = np.vstack(tmp)
qqq_trn1 = qqq_trn[:,::-1,:,:]
qqq_trn2 = qqq_trn[:,:,::-1,:]
qqq_trn4 = np.transpose(qqq_trn,[0,2,1,3])
qqq_trn3 = qqq_trn[:,::-1,::-1,:]
qqq_trn5 = qqq_trn4[:,::-1,:,:]
qqq_trn6 = qqq_trn4[:,:,::-1,:]
qqq_trn7 = qqq_trn3[:,::-1,::-1,:]

yyy_trn = tmpy
nn,ny,nx,nl = qqq_trn.shape
print('nn ny nx nl',nn,ny,nx,nl)
exec(open('tensorflow_ae_stage1.py').read())
exec(open('tensorflow_ae_stage2.py').read())
exec(open('tensorflow_ae_stage3.py').read())
exec(open('tensorflow_classify_stage4.py').read())

tf_input = tf.placeholder(tf.float32, [ni,ny,nx,nl])
tf_yyy = tf.placeholder(tf.int64, [ni])
tf_encode1 = get_encode1(tf_input)
tf_encode2 = get_encode2(tf_encode1)
tf_encode3 = get_encode3(tf_encode2)

# sess.run(tf.initialize_all_variables())

# nj = 9
# tmp = []
# tmpy = []
# for aa in range(nj):
#     tmpx = []
#     iii = random.sample(range(ns),ni)
#     for aa in range(ni):
#         ii = iii[aa]
#         path_data = data_table[ii][0]
#         img_tmp = Image.open(path_data,'r')#
#         tmpx.append((np.asarray(img_tmp) / 255.0)[np.newaxis,:,:,:])
#         tmpy.append(int(data_table[ii][1]))
#     qqq_trn = np.vstack(tmpx)
#     yyy_trn = tmpy
#     hoge = tf_encode3.eval({tf_input:qqq_trn})
#     tmp.append(hoge)

tf_encode4 = get_encode4(tf_encode3)

tf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(tf_encode4[:,0,0,:],tf_yyy)
tf_mean_loss  = tf.reduce_mean(tf_loss)
learning_rate = 1e-3
tf_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
tf_train = tf_optimizer.minimize(tf_loss)

sess.run(tf.initialize_all_variables())

for tt in range(tmax):
    if((tprint > 0) and (tt % tprint==0)):
        print(tt,tf_mean_loss.eval({tf_input: qqq_trn, tf_yyy: yyy_trn}))
    if((tstage > 0) and (tt % tstage==0)):
        iii = random.sample(range(ns),ni)
        tmp = []
        tmpy = []
        for aa in range(ni):
            ii = iii[aa]
            path_data = data_table[ii][0]
            img_tmp = Image.open(path_data,'r')#
            tmp.append((np.asarray(img_tmp) / 255.0)[np.newaxis,:,:,:])
            tmpy.append(int(data_table[ii][1]))
        qqq_trn = np.vstack(tmp)
        yyy_trn = tmpy
        qqq_trn1 = qqq_trn[:,::-1,:,:]
        qqq_trn2 = qqq_trn[:,:,::-1,:]
        qqq_trn4 = np.transpose(qqq_trn,[0,2,1,3])
        qqq_trn3 = qqq_trn[:,::-1,::-1,:]
        qqq_trn5 = qqq_trn4[:,::-1,:,:]
        qqq_trn6 = qqq_trn4[:,:,::-1,:]
        qqq_trn7 = qqq_trn3[:,::-1,::-1,:]
    #
    sess.run(tf_train,{tf_input: qqq_trn, tf_yyy: yyy_trn})
    sess.run(tf_train,{tf_input: qqq_trn1, tf_yyy: yyy_trn})
    sess.run(tf_train,{tf_input: qqq_trn2, tf_yyy: yyy_trn})
    sess.run(tf_train,{tf_input: qqq_trn3, tf_yyy: yyy_trn})
    sess.run(tf_train,{tf_input: qqq_trn4, tf_yyy: yyy_trn})
    sess.run(tf_train,{tf_input: qqq_trn5, tf_yyy: yyy_trn})
    sess.run(tf_train,{tf_input: qqq_trn6, tf_yyy: yyy_trn})
    sess.run(tf_train,{tf_input: qqq_trn7, tf_yyy: yyy_trn})

if(tt < tmax):
    print(tmax,tf_mean_loss.eval({tf_input: qqq_trn, tf_yyy: yyy_trn}))

hoge = tf.argmax(tf_encode4[:,0,0,:],dimension=1)
fuga = hoge.eval({tf_input:qqq_trn})
print(np.sum(fuga == yyy_trn),"/",len(fuga),"\n")
#
# save parameters
#

if(trainable1):
    save_stage1()
    print('stamp1 = \'{}\''.format(stamp))

if(trainable2):
    save_stage2()
    print('stamp2 = \'{}\''.format(stamp))

if(trainable3):
    save_stage3()
    print('stamp3 = \'{}\''.format(stamp))

if(trainable4):
    save_stage4()
    print('stamp4 = \'{}\''.format(stamp))
