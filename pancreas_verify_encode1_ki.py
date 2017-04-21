#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('verify stacked autoencoder stage 1')

import os
import sys
import csv
import numpy as np
import numpy as np
import pickle
from PIL import Image

import tensorflow as tf
import tensorflow_ae_base

from tensorflow_ae_base import *
import tensorflow_util

import myutil
from myutil import source

exec(open('extern_params.py').read())

dir_data = 'dat1'

path_dst = os.path.join(dir_data,'img1_1K.tif')
img_dst = Image.open(path_dst,'r')
qqq_dst = np.asarray(img_dst,dtype=np.float32)[np.newaxis] / 256.0
nn,ny,nx,nl = qqq_dst.shape

stype = 'ki'
# stamp1
if(stamp1=='NA'):
    assert('set stamp1')
batch_size = 1
exec(open('pancreas_net_encode1.py').read())

tf_dst = tf.placeholder(tf.float32, [batch_size,ny,nx,nl])
tf_conv1 = get_conv1(tf_dst)
tf_encode1 = get_encode1(tf_conv1)
tf_deconv1 = get_deconv1(tf_encode1)
tf_risa1 = get_risa1(tf_conv1)
tf_mean_error = tf.reduce_mean(tf.square(tf_deconv1 - tf_dst))

sess.run(tf.global_variables_initializer())

qqq_conv1 = tf_conv1.eval({tf_dst: qqq_dst})
np.sum(qqq_conv1,axis=(0,1,2))

print(tf_risa1.eval({tf_dst: qqq_dst[0:batch_size]}))
print(tf_mean_error.eval({tf_dst:qqq_dst[0:batch_size]}))

# t_1 = tf.nn.conv2d(np.array(qqq_dst[0:1,0:7,0:7,0:3],dtype=np.float32), weight1['conv'], strides=[1, 1, 1, 1], padding='SAME')
# ww_conv1 = weight1['conv'].eval()
# bb_conv1 = bias1['conv'].eval()
# qqq_tmp = np.zeros((7,7,3))
# for aa in range(7):
#     for bb in range(7):
#         for cc in range(3):
#             qqq_in = qqq_dst[0,0+aa:7+aa,0+bb:7+bb,:]
#             qqq_fl = ww_conv1[:,:,:,cc]
#             qqq_tmp[aa,bb,cc] = np.sum(np.multiply(qqq_in,qqq_fl)) #+bb_conv1[cc]

#output[b, i, j, k] =
#    sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
#                    filter[di, dj, q, k]


aa = 0
img_ww = tensorflow_util.get_image_from_ww(weight1['conv'].eval()[:,:,:,aa])
for aa in range(1,nf_conv1):
    img_ww = myutil.cbind_image(img_ww,tensorflow_util.get_image_from_ww(weight1['conv'].eval()[:,:,:,aa]))
path_img = 'out1/img_ww.{}.jpg'.format(stamp1)
img_ww.save(path_img)
print(path_img)

qqq_conv1 = tf_conv1.eval({tf_dst: qqq_dst})
img_conv1 = tensorflow_util.get_image_from_encode(qqq_conv1)
path_img = 'out1/img_conv1.{}.jpg'.format(stamp1)
img_conv1.save(path_img)
print(path_img)

qqq_encode1 = tf_encode1.eval({tf_dst: qqq_dst})
img_encode1 = tensorflow_util.get_image_from_encode(qqq_encode1)
path_img = 'out1/img_encode1.{}.jpg'.format(stamp1)
img_encode1.save(path_img)
print(path_img)

qqq_deconv1 = tf_deconv1.eval({tf_dst: qqq_dst})
img_out = tensorflow_util.get_image_from_qqq(qqq_deconv1)

img_cmp = myutil.cbind_image(img_dst,img_out)
path_img = 'out1/img_cmp1.{}.jpg'.format(stamp1)
img_cmp.save(path_img)
print(path_img)
batch_size = 32
