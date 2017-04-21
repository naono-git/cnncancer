#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('train stacked autoencoder 1')
# exec(open('pancreas_train_stage1_kie.py').read())
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

exec(open('extern_params.py').read())



#
# load sample data
#
dir_data = 'dat1'

nx = 32 # sample size
file_input = 'pancreas_ki_w{}.npy'.format(nx)
path_data = os.path.join(dir_data,file_input)
print('load input from {}'.format(path_data))
qqq_trn = np.load(path_data)

nn,ny,nx,nl = qqq_trn.shape
print('nn ny nx nl',nn,ny,nx,nl)

stype = 'ki'
# stamp1
exec(open('pancreas_net_encode1.py').read())

##
##
##
#
# setup optimizer
#
tf_input = tf.placeholder(tf.float32, [None,ny,nx,nl])
tf_conv1 = get_conv1(tf_input)
tf_encode1 = get_encode1(tf_conv1)
tf_deconv1 = get_deconv1(tf_encode1)

tf_risa1 = get_risa1(tf_conv1)

# local_entropy = get_local_entropy_encode1(tf_encode1)
# mean_entropy = tf.reduce_mean(local_entropy)
# optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
# train = optimizer.minimize(mean_error + lambda_s*mean_entropy)

## risa = tf.sqrt(tf.reduce_mean(tf.square(tf_encode1)))

tf_mean_error = tf.reduce_mean(tf.square(tf_deconv1 - tf_input))

tf_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
## train = optimizer.minimize(tf_mean_error)
tf_train = tf_optimizer.minimize(tf_mean_error + lambda_risa*tf_risa1)

sess.run(tf.global_variables_initializer())

qqq_conv1 = tf_conv1.eval({tf_input: qqq_trn[0:batch_size]})
np.sum(qqq_conv1,axis=(0,1,2))
qqq_encode1 = tf_encode1.eval({tf_input: qqq_trn[0:batch_size]})
np.sum(qqq_encode1,axis=(0,1,2))
qqq_deconv1 = tf_deconv1.eval({tf_input: qqq_trn[0:batch_size]})

if(False):
    print('overwrite conv and deconv')
    ww_tmp = weight1['conv'].eval()
    ww_tmp[:,:,:,0] = ww_tmp[:,:,:,1]+tf.truncated_normal([fs_1,fs_1,nf_RGB],stddev=0.003).eval()
    ww_tmp[:,:,:,2] = ww_tmp[:,:,:,1]+tf.truncated_normal([fs_1,fs_1,nf_RGB],stddev=0.003).eval()
    sess.run(weight1['conv'].assign(ww_tmp))

    bb_tmp = bias1['conv'].eval()
    bb_tmp[0] = bb_tmp[1]
    bb_tmp[2] = bb_tmp[1]
    sess.run(bias1['conv'].assign(bb_tmp))

    ww_tmp = weight1['deconv'].eval()
    ww_tmp[:,:,0,:] = ww_tmp[:,:,1,:]/3+tf.truncated_normal([fs_1,fs_1,nf_RGB],stddev=0.003).eval()
    ww_tmp[:,:,2,:] = ww_tmp[:,:,1,:]/3+tf.truncated_normal([fs_1,fs_1,nf_RGB],stddev=0.003).eval()
    sess.run(weight1['deconv'].assign(ww_tmp))

if(False):
    print('overwrite encode and hidden')
    ww_tmp = weight1['encode'].eval()
    ww_tmp[:,:,:,0] = ww_tmp[:,:,:,1]/3+tf.truncated_normal([fs_1,fs_1,nf_conv1],stddev=0.003).eval()
    ww_tmp[:,:,:,2] = ww_tmp[:,:,:,1]/3+tf.truncated_normal([fs_1,fs_1,nf_conv1],stddev=0.003).eval()
    sess.run(weight1['encode'].assign(ww_tmp))

    ww_tmp = weight1['hidden'].eval()
    ww_tmp[:,:,0,:] = ww_tmp[:,:,1,:]/3+tf.truncated_normal([fs_1,fs_1,nf_conv1],stddev=0.003).eval()
    ww_tmp[:,:,2,:] = ww_tmp[:,:,1,:]/3+tf.truncated_normal([fs_1,fs_1,nf_conv1],stddev=0.003).eval()
    sess.run(weight1['hidden'].assign(ww_tmp))

print("risa = ",tf_risa1.eval({tf_input:qqq_trn[0:batch_size]}))
print("mean_error = ",tf_mean_error.eval({tf_input:qqq_trn[0:batch_size]}))
#
# train loop
#
iii_bin = np.arange(batch_size,nn,batch_size)
iii_nn = np.arange(nn)
iii_batches = np.split(iii_nn,iii_bin)

for tt in range(tmax):
    if(tt % tprint==0):
        error_out = [tf_mean_error.eval({tf_input: qqq_trn[iii,]}) for iii in iii_batches]
        print(tt,np.mean(error_out))
    np.random.shuffle(iii_nn)
    iii_batches = np.split(iii_nn,iii_bin)
    for iii in iii_batches:
        sess.run(tf_train,feed_dict={tf_input: qqq_trn[iii,]})

tmp = [tf_mean_error.eval({tf_input: qqq_trn[iii,]}) for iii in iii_batches]
error_tmp = np.mean(tmp)
print(tmax,error_tmp)

#
# save parameters
#
save_params(dir_out)
save_params_txt('out1')

save_stage1(dir_out, stype='ki')

myutil.show_timestamp()
print('stamp1 = \'{}\''.format(stamp))
