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
# ss = 128
# na = 16

path_data =os.path.join(dir_data,'tcga_trn_w64.npy')
print(path_data)
qqq_tmp = np.load(path_data)
nn,ny,nx,nl = qqq_tmp.shape

tf_input = tf.placeholder(tf.float32, [None,ny,nx,nl])
tf_encode1 = get_encode1(tf_input)
sess.run(tf.initialize_all_variables())

iii_bin = np.arange(batch_size,nn,batch_size)
iii_nn = np.arange(nn)
iii_batches = np.split(iii_nn,iii_bin)

tmp = [tf_encode1.eval({tf_input: qqq_tmp[iii,]}) for iii in iii_batches]
qqq_encode1 = np.vstack(tmp)

np.save('dat1/tcga_encode1_w32.{}.npy'.format(stamp1),qqq_encode1)
