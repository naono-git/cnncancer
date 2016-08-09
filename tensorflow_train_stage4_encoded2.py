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
batch_size = 32

## one batch one file
# bronchioid
# magnoid
# squamoid	

import random

na = 27

tmpx = []
tmpy = []
for aa in range(na):
    dir_data = '/home/nono/Documents/cnncancer/out1'
    file_data = 'qqq_encode2_tcga_w512_{}.{}.npy'.format(stamp2,aa+1)
    path_data = os.path.join(dir_data,file_data)
    tmpx.append(np.load(path_data))
    file_data = 'yyy_encode2_tcga_w512_{}.{}.npy'.format(stamp2,aa+1)
    path_data = os.path.join(dir_data,file_data)
    tmpy.append(np.load(path_data))
tmp = np.vstack(tmpx)
n0 = tmp.shape[0]

iii_trn = np.array(random.sample(range(n0),256))
iii_vld = [ xx for ii,xx in enumerate(np.arange(n0)) if xx not in iii_trn ]

qqq_trn = tmp[iii_trn,]
qqq_vld = tmp[iii_vld,]

qqq_trn1 = qqq_trn[:,::-1,:,:]
qqq_trn2 = qqq_trn[:,:,::-1,:]
qqq_trn4 = np.transpose(qqq_trn,[0,2,1,3])
qqq_trn3 = qqq_trn[:,::-1,::-1,:]
qqq_trn5 = qqq_trn4[:,::-1,:,:]
qqq_trn6 = qqq_trn4[:,:,::-1,:]
qqq_trn7 = qqq_trn3[:,::-1,::-1,:]

yyy = np.vstack(tmpy)[:,1]
yyy_trn = yyy[iii_trn]
yyy_vld = yyy[iii_vld]
nn,ny,nx,nl = qqq_trn.shape
print('nn ny nx nl',nn,ny,nx,nl)
exec(open('tensorflow_ae_stage1.py').read())
exec(open('tensorflow_ae_stage2.py').read())
exec(open('tensorflow_ae_stage3.py').read())
exec(open('tensorflow_classify_stage4.py').read())

tf_encode2 = tf.placeholder(tf.float32, [None,ny,nx,nf_encode2])
tf_yyy = tf.placeholder(tf.int64, [None])
tf_encode3 = get_encode3(tf_encode2)
tf_encode4 = get_encode4(tf_encode3)

tf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(tf_encode4[:,0,0,:],tf_yyy)
tf_mean_loss  = tf.reduce_mean(tf_loss)
learning_rate = 1e-3
tf_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
tf_train = tf_optimizer.minimize(tf_loss)

sess.run(tf.initialize_all_variables())

iii_bin = np.arange(batch_size,nn,batch_size)
iii_nn = np.arange(nn)
iii_batches = np.split(iii_nn,iii_bin)

for tt in range(tmax):
    if((tprint > 0) and (tt % tprint==0)):
        tmp = [tf_loss.eval({tf_encode2:qqq_trn[iii,],tf_yyy:yyy_trn[iii]})]
        print(tt,np.mean(tmp))
    #
    np.random.shuffle(iii_nn)
    iii_batches = np.split(iii_nn,iii_bin)
    #
    for iii in iii_batches:
        sess.run(tf_train,{tf_encode2: qqq_trn[iii,],  tf_yyy: yyy_trn[iii,]})
        sess.run(tf_train,{tf_encode2: qqq_trn1[iii,], tf_yyy: yyy_trn[iii,]})
        sess.run(tf_train,{tf_encode2: qqq_trn2[iii,], tf_yyy: yyy_trn[iii,]})
        sess.run(tf_train,{tf_encode2: qqq_trn3[iii,], tf_yyy: yyy_trn[iii,]})
        sess.run(tf_train,{tf_encode2: qqq_trn4[iii,], tf_yyy: yyy_trn[iii,]})
        sess.run(tf_train,{tf_encode2: qqq_trn5[iii,], tf_yyy: yyy_trn[iii,]})
        sess.run(tf_train,{tf_encode2: qqq_trn6[iii,], tf_yyy: yyy_trn[iii,]})
        sess.run(tf_train,{tf_encode2: qqq_trn7[iii,], tf_yyy: yyy_trn[iii,]})

if(tt < tmax):
    tmp = [tf_loss.eval({tf_encode2:qqq_trn[iii,],tf_yyy:yyy_trn[iii]})]
    print(tmax,np.mean(tmp))

hoge = verify_class(qqq_trn,yyy_trn)
print(np.sum(hoge[:,0]==hoge[:,1]),"/",hoge.shape[0],"\n")

fuga = verify_class(qqq_vld,yyy_vld)
print(np.sum(fuga[:,0]==fuga[:,1]),"/",fuga.shape[0],"\n")

qqq_vld1 = qqq_vld[:,::-1,:,:]

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
