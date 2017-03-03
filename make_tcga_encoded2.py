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

exec(open('extern_params.py').read())

# extern stamp1 stamp2
exec(open('tensorflow_ae_stage1.py').read())
exec(open('tensorflow_ae_stage2.py').read())

# extern 
ss = 128

path_data =os.path.join(dir_data,'tcga_trn_w128.npy')
print(path_data)
qqq_tmp = np.load(path_data)
nn,ny,nx,nl = qqq_tmp.shape

exec(open('tensorflow_ae_stage1.py').read())
exec(open('tensorflow_ae_stage2.py').read())

tf_input = tf.placeholder(tf.float32, [None,ny,nx,nl])
tf_encode1 = get_encode1(tf_input)
tf_encode2 = get_encode2(tf_encode1)
sess.run(tf.initialize_all_variables())

iii_bin = np.arange(batch_size,nn,batch_size)
iii_nn = np.arange(nn)
iii_batches = np.split(iii_nn,iii_bin)

tmp = [tf_encode2.eval({tf_input: qqq_tmp[iii,]}) for iii in iii_batches]
qqq_encode2 = np.vstack(tmp)

np.save('dat1/tcga_encode2_w32.{}.npy'.format(stamp2),qqq_encode2)
