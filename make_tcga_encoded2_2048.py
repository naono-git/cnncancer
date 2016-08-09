#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('make pre-encoded tcga data from 2048')

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

import random
exec(open('extern_params.py').read())

# extern stamp1 stamp2
exec(open('tensorflow_ae_stage1.py').read())
exec(open('tensorflow_ae_stage2.py').read())

# extern 
ss = 2048

data_table = list(csv.reader(open('filelist.txt','r'), delimiter='\t'))
ns = len(data_table)
ni = 16
na = ns // ni
tmp = []
for ii in range(ni) :
    path_data = data_table[ii][0]
    img_tmp = Image.open(path_data,'r')
    tmp.append((np.asarray(img_tmp) / 255.0)[np.newaxis,:,:,:])
qqq_trn = np.vstack(tmp)
nn,ny,nx,nl = qqq_trn.shape

exec(open('tensorflow_ae_stage1.py').read())
exec(open('tensorflow_ae_stage2.py').read())

tf_input = tf.placeholder(tf.float32, [ni,ny,nx,nl])
tf_encode1 = get_encode1(tf_input)
tf_encode2 = get_encode2(tf_encode1)
sess.run(tf.initialize_all_variables())

iii_rand = np.array(random.sample(range(ns),ns))
for aa in range(na) :
    print(aa,)
    iii = iii_rand[np.arange(16)+aa*ni]
    tmpx = []
    tmpy = []
    for ii in iii :
        tmp = []
        path_data = data_table[ii][0]
        img_tmp = Image.open(path_data,'r')
        tmpx.append((np.asarray(img_tmp) / 256.0)[np.newaxis,:,:,:])
        tmpy.append([ii,int(data_table[ii][1])])
    qqq_trn = np.vstack(tmpx)
    yyy_trn = np.vstack(tmpy)
    qqq_encode2 = tf_encode2.eval({tf_input: qqq_trn})
    ww_enc = qqq_encode2.shape[2]
    np.save('out1/qqq_encode2_tcga_w{}_{}.{}.npy'.format(ww_enc,stamp2,aa+1),qqq_encode2)
    np.save('out1/yyy_encode2_tcga_w{}_{}.{}.npy'.format(ww_enc,stamp2,aa+1),yyy_trn)
