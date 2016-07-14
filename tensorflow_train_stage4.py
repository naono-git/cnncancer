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
path_data3 = ('/Users/nono/Documents/data/tissue_images/TCGA-05-4384-01A-01-BS1_2048_files/15/3_6.jpeg',
              '/Users/nono/Documents/data/tissue_images/TCGA-38-4631-11A-01-TS1_2048_files/14/4_3.jpeg', 
              '/Users/nono/Documents/data/tissue_images/TCGA-05-4425-01A-01-BS1_2048_files/15/7_6.jpeg')

import random

data_table = list(csv.reader(open('/Users/nono/Documents/cnncancer/filelist.txt','r'), delimiter='\t'))
ns = len(data_table)


ni = 9
iii = random.sample(range(ns),ni)
## iii = [120, 43, 82, 128, 139, 71, 96, 39, 59]
tmp = []
tmpy = []
for aa in range(ni):
    path_data = data_table[aa][0]
    img_tmp = Image.open(path_data,'r')
    tmp.append((np.asarray(img_tmp) / 255.0)[np.newaxis,:,:,:])
    tmpy.append(int(data_table[aa][1]))
qqq_trn = np.vstack(tmp)
nn,ny,nx,nl = qqq_trn.shape
print('nn ny nx nl',nn,ny,nx,nl)
exec(open('tensorflow_ae_stage1.py').read())
exec(open('tensorflow_ae_stage2.py').read())
exec(open('tensorflow_ae_stage3.py').read())
exec(open('tensorflow_classify_stage4.py').read())

ni = 9

tf_input = tf.placeholder(tf.float32, [ni,ny,nx,nl])
tf_yyy   = tf.placeholder(tf.int32, [ni])
tf_encode1 = get_encode1(tf_input)
tf_encode2 = get_encode2(tf_encode1)
tf_encode3 = get_encode3(tf_encode2)
tf_encode4 = get_encode4(tf_encode3)

tf_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(tf_encode4[:,0,0,:],tf_yyy)
tf_mean_loss  = tf.reduce_mean(tf_loss)
tf_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
tf_train = tf_optimizer.minimize(tf_loss)

sess.run(tf.initialize_all_variables())

for tt in range(tmax):
    iii = random.sample(range(ns),ni)
    tmp = []
    tmpy = []
    for aa in range(ni):
        path_data = data_table[aa][0]
        img_tmp = Image.open(path_data,'r')
        tmp.append((np.asarray(img_tmp) / 255.0)[np.newaxis,:,:,:])
        tmpy.append(int(data_table[aa][1]))
    qqq_trn = np.vstack(tmp)
    yyy_trn = tmpy
    #
    if((tprint > 0) and (tt % tprint==0)):
        print(tt,tf_loss.eval({tf_input: qqq_trn, tf_yyy: yyy_trn}))
    sess.run(tf_train,{tf_input: qqq_trn, tf_yyy: yyy_trn})

if(tt < tmax):
    print(tmax,tf_loss.eval({tf_input: qqq_trn, tf_yyy: yyy_trn}))
