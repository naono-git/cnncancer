#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('make pre-encoded tcga data')

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

# extern stamp1
exec(open('tensorflow_ae_stage1.py').read())

# extern 
ss = 64
na = 4
# ss = 128
# na = 16

for aa in range(na) :
    dir_data = os.path.join(dir_input,'input_w{}'.format(ss))
    path_data =os.path.join(dir_data,'qqq_trn_w{}_{}.npy'.format(ss,aa+1))
    print(path_data)
    qqq_tmp = np.load(path_data)
    # qqq_tmp = qqq_tmp[0:64,]
    nn,ny,nx,nl = qqq_tmp.shape

    iii_bin = np.arange(batch_size,nn,batch_size)
    iii_nn = np.arange(nn)
    iii_batches = np.split(iii_nn,iii_bin)

    tf_input = tf.placeholder(tf.float32, [None,ny,nx,nl])
    tf_encode1 = get_encode1(tf_input)
    sess.run(tf.initialize_all_variables())
    
    qqq_encode1 = np.vstack([sess.run(tf_encode1,{tf_input: qqq_tmp[iii,]}) for iii in iii_batches])
    ss = qqq_encode1.shape[2]
    np.save('out1/qqq_encode1_tcga_w{}_{}.{}.npy'.format(ss,aa+1,stamp1),qqq_encode1)
