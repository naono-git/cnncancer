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
ss = 2048
ns = 256

dir_image = '/project/hikaku_db/data/tissue_images'
dir_data = 'dat1'

nx = ss
ny = ss
nl = 3

# stamp1,stamp2
exec(open('tensorflow_ae_stage1.py').read())
exec(open('tensorflow_ae_stage2.py').read())

tf_input = tf.placeholder(tf.float32, [None,ny,nx,nl])
tf_encode1 = get_encode1(tf_input)
tf_encode2 = get_encode2(tf_encode1)
sess.run(tf.initialize_all_variables())

file_imglist = 'typelist.filterd.txt'
fileTable = list(csv.reader(open("dat1/typelist.filterd.txt",'r'), delimiter='\t'))

iii_sample = np.random.choice(range(len(fileTable)),size=ns,replace=False)

index = []
qqq_trn = []
yyy_trn = []
for aa in range(ns):
    ii = iii_sample[aa]
    file_src = fileTable[ii][0]
    path_data = os.path.join(dir_image,file_src)
    ## print(path_data)
    img_src = Image.open(path_data,'r')
    mx = img_src.size[0]
    my = img_src.size[1]
    img_tmp = Image.open(path_data,'r')
    qqq_tmp = (np.asarray(img_tmp) / 255.0)[np.newaxis,:,:,:]
    qqq_encode2 = tf_encode2.eval({tf_input: qqq_tmp})
    index.append(ii)
    qqq_trn.append(qqq_encode2)
    yyy_trn.append(fileTable[ii][1])

index = np.asarray(index)
qqq_trn = np.vstack(qqq_trn)
yyy_trn = np.asarray(yyy_trn)
np.save('dat1/tcga_encode2_w512.{}.npy'.format(stamp2),qqq_trn)
np.save('dat1/type_encode2_w512.{}.npy'.format(stamp2),yyy_trn)
np.save('dat1/index_encode2_w512.{}.npy'.format(stamp2),index)
