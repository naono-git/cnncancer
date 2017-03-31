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
qqq_dst = np.asarray(img_dst)[np.newaxis] / 256.0
nn,ny,nx,nl = qqq_dst.shape

stype = 'ki'
# stamp1
if(stamp1=='NA'):
    assert('set stamp1')
exec(open('pancreas_net_encode1.py').read())

tf_dst = tf.placeholder(tf.float32, [None,ny,nx,nl])
tf_encode1 = get_encode1(tf_dst)
tf_deconv1 = get_deconv1(tf_encode1)

mean_error = tf.reduce_mean(tf.square(tf_deconv1 - tf_dst))

sess.run(tf.global_variables_initializer())

mean_error.eval({tf_dst:qqq_dst})

qqq_encode1 = tf_encode1.eval({tf_dst: qqq_dst})
img_encode1 = tensorflow_util.get_image_from_encode(qqq_encode1)
img_encode1.save('out1/img_encode1.{}.jpg'.format(stamp1))

qqq_deconv1 = tf_deconv1.eval({tf_dst: qqq_dst})
img_out = tensorflow_util.get_image_from_qqq(qqq_deconv1)

img_cmp = myutil.cbind_image(img_dst,img_out)
img_cmp.save('out1/img_cmp1.{}.jpg'.format(stamp1))
