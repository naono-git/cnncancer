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

#
# load sample data
#
dir_data = 'dat1'

print('nn ny nx nl',nn,ny,nx,nl)
exec(open('tensorflow_ae_stage1.py').read())

qqq_src = np.asarray(img1_1K)[np.newaxis] / 255.0
nn,ny,nx,nl = qqq_src.shape
#
# setup optimizer
#
tf_src = tf.placeholder(tf.float32, [None,ny,nx,nl])
tf_encode1 = get_encode1(tf_src)
tf_deconv1 = get_deconv1(tf_encode1)
mean_error = tf.reduce_mean(tf.square(tf_deconv1 - tf_src))

sess.run(tf.global_variables_initializer())

mean_error.eval({tf_src:qqq_src})

qqq_encode1 = tf_encode1.eval({tf_src: qqq_src})
img_encode1 = tensorflow_util.get_image_from_encode(qqq_encode1)
img_encode1.save('out1/encode1.{}.jpg'.format(stamp1))
qqq_deconv1 = tf_deconv1.eval({tf_src: qqq_src})
img_out = tensorflow_util.get_image_from_qqq(qqq_deconv1)

img_cmp = myutil.cbind_image(img0_1K,img_out)
img_cmp.save('out1/cmp1.{}.jpg'.format(stamp1))

for aa in range(12):
    np.savetxt('out1/qqq_encode2_{}.txt'.format(aa+1),qqq_encode2[0,:,:,aa])
