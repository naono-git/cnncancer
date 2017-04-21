#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('setup stacked autoencoder stage1')

#
# setup parameters
#

key1 = ['conv', 'encode', 'hidden', 'deconv']

# extern stamp1,trainable1
sd0 = 0.2
if(stamp1=='NA'):
    print('initialize w1 and b1 randomly. sd0 = ',sd0)
    weight1 = {
        'conv':   tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_RGB,nf_conv1],stddev=sd0),
                               trainable=trainable1,name='weight.conv1'),
        'encode': tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_conv1,nf_encode1],stddev=sd0),
                               trainable=trainable1,name='weight.encode1'),
        'hidden': tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_encode1,nf_conv1],stddev=sd0),
                               trainable=trainable1,name='weight.hidden1'),
        'deconv': tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_conv1,nf_RGB],stddev=sd0),
                               trainable=trainable1,name='weight.deconv1'),
    }
    bias1 = {
        'conv':   tf.Variable(tf.zeros([nf_conv1]),trainable=trainable1,name='conv1.bb'),
        'encode': tf.Variable(tf.zeros([nf_encode1]),trainable=trainable1),
        'hidden': tf.Variable(tf.zeros([nf_conv1]),trainable=trainable1),
        'deconv': tf.Variable(tf.zeros([nf_RGB]),trainable=trainable1),
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

def get_conv1(t_input):
    t_1 = tf.nn.conv2d(t_input, weight1['conv'], strides=[1, 1, 1, 1], padding='SAME')
    t_2 = tf.nn.bias_add(t_1, bias1['conv'])
    t_conv = tf.nn.relu(t_2)
    return(t_conv)

seg12_12= tf.constant([0,1,2,3,4,5,6,7,8,9,10,11])
seg6_12 = tf.constant([0,0,1,1,2,2,3,3,4,4,5,5])
seg4_12 = tf.constant([0,0,0,1,1,1,2,2,2,3,3,3])
seg3_12 = tf.constant([0,0,0,0,1,1,1,1,2,2,2,2])
seg2_12 = tf.constant([0,0,0,0,0,0,1,1,1,1,1,1])
seg1_12 = tf.constant([0,0,0,0,0,0,0,0,0,0,0,0])

seg6_6 = tf.constant([0,1,2,3,4,5])
seg3_6 = tf.constant([0,0,1,1,2,2])
seg2_6 = tf.constant([0,0,0,1,1,1])
seg1_6 = tf.constant([0,0,0,0,0,0])

seg3_3 = tf.constant([0,1,2])
seg1_3 = tf.constant([0,0,0])


def get_risa1(t_conv):
    t_1 = tf.square(t_conv)
    t_2 = tf.transpose(t_1)
    t_3 = tf.segment_sum(t_2, seg1_3)
    t_4 = tf.sqrt(t_3)
    t_risa = tf.reduce_mean(t_4)
    return(t_risa)

def get_encode1(t_conv):
    t_pool   = max_pool(t_conv, kk=pool_size)
    t_drop   = tf.nn.dropout(t_pool, keep_prob_1)
    t_encode = tf.nn.relu(conv2d(t_drop,  weight1['encode'], bias1['encode']))
    return(t_encode)

def get_deconv1(t_encode):
    t_tmp = t_encode
    ## t_tmp = tf.nn.relu(conv2d(t_encode, weight1['hidden'], bias1['hidden']))
    ## t_tmp = un_pool(t_tmp, kk=pool_size)
    ## t_tmp = tf.nn.relu(conv2d(t_tmp, weight1['deconv'], bias1['deconv']))
    t_tmp = tf.nn.conv2d(t_tmp, weight1['hidden'], 
                         ## output_shape=[batch_size, ny//2, nx//2, nf_conv1], 
                         strides=[1, 1, 1, 1], padding='SAME')
    t_tmp = tf.nn.bias_add(t_encode, bias1['hidden'])
    t_tmp = tf.nn.relu(t_tmp)

    t_tmp = un_pool(t_tmp, kk=pool_size)

    ##t_tmp = tf.nn.conv2d_transpose(t_tmp, weight1['conv'], 
    ##                               output_shape=[batch_size, ny, nx, nf_RGB],
    ##                               strides=[1, 1, 1, 1], padding='SAME')
    t_tmp = tf.nn.conv2d(t_tmp, weight1['deconv'], 
                         strides=[1, 1, 1, 1], padding='SAME')
    t_tmp = tf.nn.bias_add(t_tmp, bias1['deconv'])
    t_tmp = tf.nn.relu(t_tmp)
    return(t_tmp)

#
# entropy function
#

def get_local_entropy_encode1(t_encode):
    t_ooo = tf.reduce_sum(t_encode, 3, keep_dims=True)
    t_rrr = t_encode / (tf.tile(t_ooo,[1,1,1,nf_encode1])+1e-16)
    t_tmp = tf.reduce_sum(t_rrr * (-tf.log(t_rrr+1e-16)),3)
    return(t_tmp)

#
# save network parameters
#
def save_stage1(dirname=dir_out, stype=''):
    weight1_fin = {k:sess.run(v) for k,v in weight1.items()}
    bias1_fin = {k:sess.run(v) for k,v, in bias1.items()}
    if(stype==''):
        myutil.saveObject(weight1_fin,'weight1.{}.pkl'.format(stamp),dirname=dirname)
        myutil.saveObject(bias1_fin,'bias1.{}.pkl'.format(stamp),dirname=dirname)
    else:
        myutil.saveObject(weight1_fin,'weight1.{}.{}.pkl'.format(stype,stamp),dirname=dirname)
        myutil.saveObject(bias1_fin,'bias1.{}.{}.pkl'.format(stype,stamp),dirname=dirname)
    return([weight1_fin,bias1_fin])


