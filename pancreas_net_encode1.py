#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('setup stacked autoencoder stage1')

#
# setup parameters
#

# extern stamp1,trainable1
if(stamp1=='NA'):
    sd = 0.4
    print('initialize w1 and b1 randomly. sd = ',sd)
    weight1 = {
        'conv1':   tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_RGB,nf_conv1],stddev=sd),
                               trainable=trainable1,name='weight.conv1'),
        'encode1': tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_conv1,nf_encode1],stddev=sd),
                               trainable=trainable1,name='weight.encode1'),
        'hidden1': tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_encode1,nf_conv1],stddev=sd),
                               trainable=trainable1,name='weight.hidden1'),
        'deconv1': tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_conv1,nf_RGB],stddev=sd),
                               trainable=trainable1,name='weight.deconv1'),
    }
    bias1 = {
        'conv1':   tf.Variable(tf.zeros([nf_conv1]),trainable=trainable1,name='conv1.bb'),
        'encode1': tf.Variable(tf.zeros([nf_encode1]),trainable=trainable1),
        'hidden1': tf.Variable(tf.zeros([nf_conv1]),trainable=trainable1),
        'deconv1': tf.Variable(tf.zeros([nf_RGB]),trainable=trainable1),
    }
else:
    print('load w1 and b1 from',stamp1)
    if(stype==''):
        file_w1 = 'weight1.{}.pkl'.format(stamp1)
        file_b1 = 'bias1.{}.pkl'.format(stamp1)
    else:
        file_w1 = 'weight1.{}.{}.pkl'.format(stype,stamp1)
        file_b1 = 'bias1.{}.{}.pkl'.format(stype,stamp1)

    path_w1 = os.path.join(dir_out,file_w1)
    path_b1 = os.path.join(dir_out,file_b1)

    if(not os.path.exists(path_w1) or not os.path.exists(path_b1)):
        myutil.getRemoteFile([file_w1,file_b1],dirname='Documents/cnncancer/out1')
    
    weight1 = tensorflow_ae_base.load_tf_variable(path_w1,key1,trainable=trainable1)
    bias1   = tensorflow_ae_base.load_tf_variable(path_b1,key1,trainable=trainable1)
#end if(stamp=='NA')

#
# setup layers
#

def get_encode1(qqq):
    conv1   = tf.nn.relu(conv2d(qqq,    weight1['conv1'],  bias1['conv1']))
    pool1   = max_pool(conv1,kk=pool_size)
    encode1 = tf.nn.relu(conv2d(pool1,  weight1['encode1'],bias1['encode1']))
    return(encode1)

def get_deconv1(qqq):
    hidden1 = tf.nn.relu(conv2d(qqq,    weight1['hidden1'],bias1['hidden1']))
    unpool1 = un_pool(hidden1,kk=pool_size)
    deconv1 = tf.nn.relu(conv2d(unpool1,weight1['deconv1'],bias1['deconv1']))
    return(deconv1)

#
# entropy function
#

def get_local_entropy_encode1(qqq):
    ooo = tf.reduce_sum(qqq,3,keep_dims=True)
    rrr = qqq / (tf.tile(ooo,[1,1,1,nf_encode1])+1e-16)
    tmp = tf.reduce_sum(rrr * (-tf.log(rrr+1e-16)),3)
    return(tmp)

#
# save network parameters
#
def save_stage1(stype=''):
    weight1_fin = {k:sess.run(v) for k,v in weight1.items()}
    bias1_fin = {k:sess.run(v) for k,v, in bias1.items()}
    if(stype==''):
        myutil.saveObject(weight1_fin,'weight1.{}.pkl'.format(stamp),dirname=dir_out)
        myutil.saveObject(bias1_fin,'bias1.{}.pkl'.format(stamp),dirname=dir_out)
    else:
        myutil.saveObject(weight1_fin,'weight1.{}.{}.pkl'.format(stype,stamp),dirname=dir_out)
        myutil.saveObject(bias1_fin,'bias1.{}.{}.pkl'.format(stype,stamp),dirname=dir_out)
    return([weight1_fin,bias1_fin])


