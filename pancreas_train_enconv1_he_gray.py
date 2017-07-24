#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('train stacked autoencoder stage 1')

import os
import sys
import re
import csv
import numpy as np
import pickle
from PIL import Image

import tensorflow as tf
import sonnet as snt

import myutil
import tensorflow_util

import configparser

## time stamp
stamp = myutil.show_timestamp()
print('stamp = ',stamp)

configparser = configparser.ConfigParser()   
configparser.read('par1/par1.txt')

# from extern_params import  *

dir_data = configparser.get('train', 'dir_data')[1:-1]
dir_out = configparser.get('train', 'dir_out')[1:-1]

if(not 'stamp1' in globals()):
    stamp1 = configparser.get('train', 'stamp1')[1:-1]
if(stamp1=='NA'):
    stamp1 = stamp
    print('new run stamp1 = ', stamp1)
else:
    print('continue stamp1 = ',stamp1)
configparser.set('train', 'stamp1', stamp1)

random_seed = int(stamp1) % 4294967295
np.random.seed(random_seed)

if(not 'nt_max' in globals()):
    nt_max = configparser.getint('train', 'nt_max')
if(not 'nt_print' in globals()):
    nt_print = configparser.getint('train', 'nt_print')
if(not 'learning_rate' in globals()):
    learning_rate = configparser.getfloat('train', 'learning_rate')
if(not 'batch_size' in globals()):
    batch_size = configparser.getint('train', 'batch_size')
if(not 'sd0' in globals()):
    sd0      = configparser.getfloat('train', 'sd0')

#
# setup tensorflow session
#
if(not 'sess' in locals()):
    print('create a new interactive session')
    sess = tf.InteractiveSession()
else:
    tf.reset_default_graph()
    sess = tf.InteractiveSession()

if(stamp1 == 'NA'):
    # read meta parameters
    stamp1 = stamp
    dir_run = os.path.join(dir_out,'run.{}'.format(stamp1))
    os.makedirs(dir_run,exist_ok=True)
    # write meta parameters to
    # out1/run.{stamp1}/par.{stamp1}.txt
else:
    dir_run = os.path.join(dir_out,'run.{}'.format(stamp1))

#
# load sample data
#

nx = 32 # sample size

file_trn = 'pancreas_he_w{}.npy'.format(nx)
path_trn = os.path.join(dir_data,file_trn)
print('load input from {}'.format(path_trn))
qqq_trn = np.load(path_trn)
qqq_trn = np.mean(qqq_trn,axis=3,keepdims=True)

nn,ny,nx,nl = qqq_trn.shape
print('nn ny nx nl',nn,ny,nx,nl)

sid        = configparser.get('net', 'sid')[1:-1]
nf_RGB     = 3
nf_input   = configparser.getint('net', 'nf_input')
nf_enconv1 = configparser.getint('net', 'nf_enconv1')
px_f1      = configparser.getint('net', 'px_f1')
px_pool    = configparser.getint('net', 'px_pool')

#
# setup 
#
def max_pool(xxx, kk):
    return tf.nn.max_pool(xxx, ksize=[1,kk,kk,1], strides=[1,kk,kk,1], padding='SAME')

def un_pool(xxx, kk):
    ##ddd = xxx.get_shape().dims
    ##d1 = ddd[1].value
    ##d2 = ddd[2].value
    ##d3 = ddd[3].value
    d1 = 16
    d2 = 16
    d3 = nf_enconv1
    yyy0 = tf.stack([xxx,xxx])
    zzz0 = tf.transpose(yyy0,[1,2,0,3,4])
    xxx1 = tf.reshape(zzz0,[-1,d1*2,d2,d3])
    yyy1 = tf.stack([xxx1,xxx1])
    zzz1 = tf.transpose(yyy1,[1,2,3,0,4])
    xxx2 = tf.reshape(zzz1,[-1,d1*2,d2*2,d3])
    return xxx2

## sd0 = 0.2
## sd0 = 0.35355339059327379
## sd0 = 0.707 can't learn

with tf.variable_scope("enconv1"):
    weight = tf.get_variable('weight', [px_f1, px_f1, nf_input, nf_enconv1], initializer = tf.random_normal_initializer(0,sd0))
    bias   = tf.get_variable('bias',   [nf_enconv1], initializer = tf.constant_initializer(0))

with tf.variable_scope("deconv1"):
    weight = tf.get_variable('weight', [px_f1, px_f1, nf_enconv1, nf_input], initializer = tf.random_normal_initializer(0,sd0))
    bias   = tf.get_variable('bias',   [nf_input], initializer = tf.constant_initializer(0))

def get_enconv1(tf_input):
    with tf.variable_scope("enconv1", reuse=True):
        weight = tf.get_variable('weight')
        bias = tf.get_variable('bias')
        tf_1 = tf.nn.conv2d(tf_input, weight, strides=[1, 1, 1, 1], padding='SAME')
        tf_2 = tf.nn.bias_add(tf_1, bias)
        tf_3 = tf.nn.relu(tf_2)
        tf_4 = max_pool(tf_3, px_pool)
    return(tf_4)

def get_deconv1(tf_enconv1):
    with tf.variable_scope("deconv1", reuse=True):
        weight = tf.get_variable('weight')
        bias = tf.get_variable('bias')
        tf_1 = un_pool(tf_enconv1, px_pool)
        tf_2 = tf.nn.conv2d(tf_1, weight, strides=[1, 1, 1, 1], padding='SAME')
        tf_3 = tf.nn.bias_add(tf_2, bias)
        tf_4 = tf.nn.relu(tf_3)
    return(tf_4)

def get_deconv_layers(qqq_enconv1):
    nc = qqq_enconv1.shape[0]
    qqq_deconv_layers = np.zeros((1,ny*nf_enconv1,nx*nc,nf_RGB))
    for dd in range(nf_enconv1):
        qqq_tmp = qqq_enconv1*0
        qqq_tmp[:,:,:,dd] = qqq_enconv1[:,:,:,dd]
        qqq_deconv1 = get_deconv1(qqq_tmp).eval()
        for cc in range(nc):
            qqq_deconv_layers[0,ny*dd:(ny*dd+ny),nx*cc:(nx*cc+nx),:] = qqq_deconv1[cc,]
    return(qqq_deconv_layers)

tf_tg = tf.Variable(0, name='global_step', trainable=False)

tf_input = tf.placeholder(tf.float32, [None,ny,nx,nl])
tf_enconv1 = get_enconv1(tf_input)
tf_mean_enconv1 = tf.reduce_mean(tf_enconv1,axis=(0,1,2))
tf_deconv1 = get_deconv1(tf_enconv1)
    
## tf_mean_error = tf.reduce_mean(tf.square(tf_deconv1 - tf_input))
tf_loss = tf.norm(tf_deconv1 - tf_input,ord=1)
tf_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
tf_train = tf_optimizer.minimize(tf_loss)

# batch 
train_batch = tf.train.batch([qqq_trn],batch_size,enqueue_many=True)
tf.train.start_queue_runners(sess)

# saver
# var_conv1= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='enconv1')
# var_conv2= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='deconv1')
tf_saver = tf.train.Saver()
# tf_saver1 = tf.train.Saver(var_conv1)
# tf_saver2 = tf.train.Saver(var_conv2)
path_save = os.path.join(dir_run,"model.{}".format(stamp1))

# create a new Summary object with your measure
tf_summary = tf.Summary()
tf_summary_writer = tf.summary.FileWriter(dir_run, sess.graph)

# merge summary
# tf_ph_error = tf.placeholder(tf.float32, (1))
tf.summary.scalar("error",tf_loss)
tf.summary.histogram("mean_enconv1",tf_mean_enconv1)
#tf.summary.scalar('conv1_0',tf_enconv1_mean_0)
#tf.summary.scalar('conv1_1',tf_enconv1_mean_1)
#tf.summary.scalar('conv1_2',tf_enconv1_mean_2)
tf_summary_merged = tf.summary.merge_all()

ckpt = tf.train.get_checkpoint_state(dir_run)
if(ckpt):
    path_model = ckpt.model_checkpoint_path
else:
    path_model = 'NA'

print(stamp1)

if(re.match('.*{}.*'.format(stamp1),path_model)):
    print('load w1 and b1 from',stamp1)
    tf_saver.restore(sess,path_model)
    #tf_saver1.restore(sess,path_model)
    #tf_saver2.restore(sess,path_model)
else:
    print('initialize w1 and b1 randomly. sd0 = ',sd0)
    sess.run(tf.global_variables_initializer())

if(False):
    with tf.variable_scope("enconv1", reuse=True):
        weight = tf.get_variable('weight')
        ##tf.assign(weight)
        bias = tf.get_variable('bias')
        ##tf.assign(bias) = zero


with open(os.path.join(dir_run,"par.{}.txt".format(stamp1)), 'w') as file_cnf:
    configparser.write(file_cnf)


#
# train loop
#

nb = (qqq_trn.shape[0] // batch_size)

for tt in range(nt_max):
    if(tt % nt_print==0):
        tf_saver.save(sess, path_save, global_step=tf_tg.eval())
        error_bb = np.zeros(nb)
        for aa in range(nb):
            qqq_tmp = sess.run(train_batch)
            error_bb[aa] = tf_loss.eval({tf_input: qqq_tmp})
        error_mean = np.mean(error_bb)
        print(tf_tg.eval(),error_mean)

        tf_summary_writer.add_summary(tf_summary_merged.eval({tf_input: qqq_tmp}), tf_tg.eval())
        tf_summary_writer.flush()

    for aa in range(nb):
        qqq_tmp = sess.run(train_batch)
        sess.run(tf_train,{tf_input: qqq_tmp})
    sess.run(tf_tg.assign_add(1))


if(tf_tg.eval() % nt_print == 0):
    error_bb = np.zeros(nb)
    for aa in range(nb):
        qqq_tmp = sess.run(train_batch)
        error_bb[aa] = tf_loss.eval({tf_input: qqq_tmp})
    error_mean = np.mean(error_bb)
    print(tf_tg.eval(),error_mean)

#
# save 
#

tf_saver.save(sess, path_save, global_step=tf_tg.eval())
#tf_saver1.save(sess, os.path.join(dir_run,"enconv1"), global_step=tf_tg.eval())
#tf_saver2.save(sess, os.path.join(dir_run,"deconv1"), global_step=tf_tg.eval())

c0 = 0
nc = 4
qqq_enconv1 = get_enconv1(qqq_trn[c0:(c0+nc),]).eval()
qqq_deconv1_layers = get_deconv_layers(qqq_enconv1)
#img_deconv1_layers = tensorflow_util.get_image_from_qqq(qqq_deconv1_layers)
tf_summary_image_deconv1_layers = tf.summary.image('deconv1_layers',qqq_deconv1_layers)
tf_summary_writer.add_summary(tf_summary_image_deconv1_layers.eval(),  tf_tg.eval())

qqq_deconv1 = tf_deconv1.eval({tf_input: qqq_trn[c0:(c0+nc),]})
qqq_deconv1_cmp = np.zeros((1,ny*2,nx*nc,nf_RGB))
for cc in range(nc):
    qqq_deconv1_cmp[0,0:ny,   nx*cc:(nx*cc+nx),] = qqq_trn[c0+cc,]
    qqq_deconv1_cmp[0,(ny):(ny+ny),nx*cc:(nx*cc+nx),] = qqq_deconv1[cc,]

tf_summary_writer.add_summary(tf.summary.image('deconv1',qqq_deconv1_cmp).eval(), tf_tg.eval())

tf_summary_writer.flush()
myutil.show_timestamp()
print('stamp1 = \'{}\''.format(stamp1))
