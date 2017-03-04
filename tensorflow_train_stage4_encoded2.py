#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('train classyfiler stage 4 from enc 2')


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

## ss = 2048 # sample size
ss = 1024
wx = 256
## one batch one file
# bronchioid
# magnoid
# squamoid	

tmpx = []
tmpy = []

dir_data = 'dat1'
file_data = 'tcga_encode2_w{}.{}.npy'.format(wx,stamp2)
path_data = os.path.join(dir_data,file_data)
print(path_data)

qqq_trn = np.load(path_data)

qqq_trn1 = qqq_trn[:,::-1,:,:]
qqq_trn2 = qqq_trn[:,:,::-1,:]
qqq_trn4 = np.transpose(qqq_trn,[0,2,1,3])
qqq_trn3 = qqq_trn[:,::-1,::-1,:]
qqq_trn5 = qqq_trn4[:,::-1,:,:]
qqq_trn6 = qqq_trn4[:,:,::-1,:]
qqq_trn7 = qqq_trn3[:,::-1,::-1,:]

nn,ny,nx,nl = qqq_trn.shape
print('nn ny nx nl',nn,ny,nx,nl)

file_data = 'type_encode2_w{}.{}.npy'.format(wx,stamp2)
path_data = os.path.join(dir_data,file_data)
type_trn = np.load(path_data)
yyy_trn = np.zeros(nn)
for ii in range(nn):
    if(type_trn[ii]=='Proximal Inflammatory'):
        yyy_trn[ii]=0
    if(type_trn[ii]=='Proximal Proliferative'):
        yyy_trn[ii]=1
    if(type_trn[ii]=='TRU'):
        yyy_trn[ii]=2

exec(open('tensorflow_ae_stage1.py').read())
exec(open('tensorflow_ae_stage2.py').read())
exec(open('tensorflow_ae_stage3.py').read())
exec(open('tensorflow_classify_stage4.py').read())

tf_encode2 = tf.placeholder(tf.float32, [None,ny,nx,nf_encode2])
tf_yyy = tf.placeholder(tf.int64, [None])
tf_encode3 = get_encode3(tf_encode2)
tf_encode4 = get_encode4(tf_encode3)

tf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tf_encode4[:,0,0,:],labels=tf_yyy)
tf_mean_loss  = tf.reduce_mean(tf_loss)
tf_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
tf_train = tf_optimizer.minimize(tf_loss)

sess.run(tf.initialize_all_variables())

batch_size = 16
nn = nn //batch_size*batch_size
iii_bin = np.arange(batch_size,nn,batch_size)
iii_nn = np.arange(nn)
iii_batches = np.split(iii_nn,iii_bin)

for tt in range(tmax):
    if((tprint > 0) and (tt % tprint==0)):
        tmp = [tf_loss.eval({tf_encode2:qqq_trn[iii],tf_yyy:yyy_trn[iii]}) for iii in iii_batches]
        print(tt,np.mean(tmp))
    #
    np.random.shuffle(iii_nn)
    iii_batches = np.split(iii_nn,iii_bin)
    #
    for iii in iii_batches:
        sess.run(tf_train,{tf_encode2: qqq_trn[iii],  tf_yyy: yyy_trn[iii]})
        sess.run(tf_train,{tf_encode2: qqq_trn1[iii], tf_yyy: yyy_trn[iii]})
        sess.run(tf_train,{tf_encode2: qqq_trn2[iii], tf_yyy: yyy_trn[iii]})
        sess.run(tf_train,{tf_encode2: qqq_trn3[iii], tf_yyy: yyy_trn[iii]})
        sess.run(tf_train,{tf_encode2: qqq_trn4[iii], tf_yyy: yyy_trn[iii]})
        sess.run(tf_train,{tf_encode2: qqq_trn5[iii], tf_yyy: yyy_trn[iii]})
        sess.run(tf_train,{tf_encode2: qqq_trn6[iii], tf_yyy: yyy_trn[iii]})
        sess.run(tf_train,{tf_encode2: qqq_trn7[iii], tf_yyy: yyy_trn[iii]})

if(tt < tmax):
    tmp = [tf_loss.eval({tf_encode2:qqq_trn[iii],tf_yyy:yyy_trn[iii]}) for iii in iii_batches]
    print(tmax,np.mean(tmp))

iii_nn = np.arange(nn)
iii_batches = np.split(iii_nn,iii_bin)
hoge = [tf_encode4.eval({tf_encode2:qqq_trn[iii]}) for iii in iii_batches]
fuga = np.reshape(np.asarray(hoge),(nn,3))
predict = np.zeros(nn)
precision = np.zeros(nn)
confusion = np.zeros((3,3))
for aa in range(nn):
    predict[aa] = np.argmax(fuga[aa,])
    precision[aa] = predict[aa]==yyy_trn[aa]
for aa in range(3):
    for bb in range(3):
        confusion[aa][bb] = np.sum(np.logical_and(yyy_trn == aa, predict==bb))

# hoge = verify_class(qqq_trn,yyy_trn)
# print(np.sum(hoge[:,0]==hoge[:,1]),"/",hoge.shape[0],"\n")

# fuga = verify_class(qqq_vld,yyy_vld)
# print(np.sum(fuga[:,0]==fuga[:,1]),"/",fuga.shape[0],"\n")

# qqq_vld1 = qqq_vld[:,::-1,:,:]

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
