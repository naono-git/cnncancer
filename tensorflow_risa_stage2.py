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

exec(open('extern_params.py').read())

#
# load input data
#
ss = 32 # sample size
if(not 'qqq_trn' in locals()):
    file_input = 'qqq_trn_w{}.npy'.format(ss)
    path_data = os.path.join(dir_input,'input_w{}'.format(ss),file_input)
    print('load input from {}'.format(path_data))
    qqq_trn = np.load(path_data)

nn,ny,nx,nl = qqq_trn.shape
print('nn ny nx nl',nn,ny,nx,nl)

nf_RGB = 3
lambda_s = 1e+4
nf_risa1 = 24
fs_1 = 8
fs_2 = 4
nf_risa2 = 96


if(stamp1=='NA'):
    ww1 = tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_RGB,nf_risa1],stddev=0.05))
else:
    file_ww1 = 'ww1_risa.{}.pkl'.format(stamp1)
    path_ww1 = os.path.join('out1',file_ww1)
    tmp = pickle.load(open(path_ww1,'rb'))
    ww1 = tf.Variable(tmp,trainable=False)

if(stamp2=='NA'):
    ww2 = tf.Variable(tf.truncated_normal([fs_2,fs_2,nf_risa1,nf_risa2],stddev=0.05))
else:
    file_ww2 = 'ww2_risa.{}.pkl'.format(stamp2)
    path_ww2 = os.path.join('out1',file_ww2)
    tmp = pickle.load(open(path_ww2,'rb'))
    ww2 = tf.Variable(tmp,trainable=trainable2)


tf_input = tf.placeholder(tf.float32, [None,ny,nx,nl])

tf_conv1 = tf.nn.conv2d(tf_input,ww1,strides=[1,8,8,1],padding='VALID')
tf_conv2 = tf.nn.conv2d(tf_conv1,ww2,strides=[1,1,1,1],padding='VALID')
tf_deconv2 = tf.nn.conv2d_transpose(value=tf_conv2,filter=ww2,output_shape=[batch_size,4,4,nf_risa1],strides=[1,1,1,1],padding='VALID')
tf_deconv1 = tf.nn.conv2d_transpose(value=tf_deconv2,filter=ww1,output_shape=[batch_size,ny,nx,nl],strides=[1,8,8,1],padding='VALID')

tf_error = tf.reduce_mean(tf.square(tf_deconv1 - tf_input))

qqq = tf.square(tf_conv2)
ooo = tf.reduce_sum(qqq,3,keep_dims=True)
rrr = qqq / (tf.tile(ooo,[1,1,1,nf_risa2])+1e-16)
tf_local_entropy = tf.reduce_sum(rrr * (-tf.log(rrr+1e-16)),3)
tf_entropy = tf.reduce_mean(tf_local_entropy)
                
# tf_score = tf_error #* lambda_s + tf_sparce1
# tf_score = tf_error * lambda_s + tf_sparce1
tf_score = lambda_s * tf_error + tf_entropy

optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(tf_score)

sess.run(tf.initialize_all_variables())

iii_bin = np.arange(batch_size,nn,batch_size)
iii_nn = np.arange(nn)
iii_batches = np.split(iii_nn,iii_bin)

for tt in range(tmax):
    if(tt % tprint==0):
        tmp = [sess.run((tf_error,tf_entropy,tf_score),{tf_input: qqq_trn[iii,]}) for iii in iii_batches]
        error_out = np.mean([xxx[0] for xxx in tmp])
        entro_out = np.mean([xxx[1] for xxx in tmp])
        score_out = np.mean([xxx[2] for xxx in tmp])
        print('tt {} error {} entropy {}  score {}'.format(tt,error_out,entro_out,score_out))
    np.random.shuffle(iii_nn)
    iii_batches = np.split(iii_nn,iii_bin)
    for iii in iii_batches:
        sess.run(train,feed_dict={tf_input: qqq_trn[iii,]})

if(tt % tprint != 0):
    tmp = [sess.run((tf_error,tf_entropy,tf_score),{tf_input: qqq_trn[iii,]}) for iii in iii_batches]
    error_out = np.mean([xxx[0] for xxx in tmp])
    entro_out = np.mean([xxx[1] for xxx in tmp])
    score_out = np.mean([xxx[2] for xxx in tmp])
    print('tt {} error {} entropy {}  score {}'.format(tt,error_out,entro_out,score_out))


ww2_out = ww2.eval()
myutil.saveObject(ww2_out,'ww2_risa.{}.pkl'.format(stamp))

myutil.timestamp()
print('stamp2 = \'{}\''.format(stamp))
