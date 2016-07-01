#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('setup stacked autoencoder stage1')

key1 = ['conva', 'convb','encode']

#
# save and load
#
def save_stage1():
    weight1_fin = {k:sess.run(v) for k,v in weight1.items()}
    myutil.saveObject(weight1_fin,'weight1.{}.pkl'.format(stamp))
    return(weight1_fin)

def load_stage1(path):
    weight1  = tensorflow_ae_base.load_tf_variable(path,key1,trainable=trainable1)
    return(weight1)

#
# setup parameters
#

# extern stamp1,trainable1
if(stamp1=='NA'):
    print('initialize w1 and b1 randomly')
    weight1 = {
        'conva' : tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_RGB,nf_conv1],stddev=0.05),trainable=trainable1,name='conv1a.ww'),
        'convb' : tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_RGB,nf_conv1],stddev=0.05),trainable=trainable1,name='conv1b.ww'),
        'encode': tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_conv1*2,nf_encode1],stddev=0.05),trainable=trainable1),
    }
else:
    print('load weight1 from',stamp1)
    file_w1 = 'weight1.{}.pkl'.format(stamp1)
    path_w1 = os.path.join('out1',file_w1)
    if(not os.path.exists(path_w1)):
        myutil.getRemoteFile(file_w1)

    weight1 = load_stage1(path_w1)
#end if(stamp1=='NA')

#
# setup layers
#

def get_encode1(qqq):
    conv1a  = tf.nn.conv2d(qqq,weight1['conva'],strides=[1,1,1,1],padding='SAME')
    pool1a  = max_pool(conv1a,kk=2)
    conv1b  = tf.nn.conv2d(qqq,weight1['convb'],strides=[1,2,2,1],padding='SAME')
    concat1 = tf.concat(3,(pool1a,conv1b))
    encode1 = tf.nn.conv2d(concat1,weight1['encode'],strides=[1,1,1,1],padding='SAME')
    relu1   = tf.nn.relu(encode1)
    return(relu1)

def get_deconv1(qqq):
    decode1  = tf.nn.conv2d_transpose(qqq,weight1['encode'],output_shape=[batch_size,ny//2,nx//2,nf_conv1*2],strides=[1,1,1,1],padding='SAME')
    unpool1a = un_pool_2(tf.slice(decode1,[0,0,0,0],[batch_size,ny//2,nx//2,nf_conv1]))
    deconv1a = tf.nn.conv2d_transpose(unpool1a,weight1['conva'],output_shape=[batch_size,ny,nx,nl],strides=[1,1,1,1],padding='SAME')
    deconv1b = tf.nn.conv2d_transpose(tf.slice(decode1,[0,0,0,3],[batch_size,ny//2,nx//2,nl]),weight1['convb'],output_shape=[batch_size,ny,nx,nl],strides=[1,2,2,1],padding='SAME')
    ## concat1  = tf.concat(3,(deconv1a,deconv1b))
    concat1  = tf.dynamic_stitch([[0,2,4],[1,3,5]], (tf.transpose(deconv1a),tf.transpose(deconv1b)))
    merge1   = tf.transpose(tf.segment_mean(concat1,[0,0,1,1,2,2]))
    relu1    = tf.nn.relu(merge1)
    return(relu1)

#
# entropy function
#

def get_local_entropy_encode1(qqq):
    ooo = tf.reduce_sum(qqq,3,keep_dims=True)
    rrr = qqq / (tf.tile(ooo,[1,1,1,nf_encode1])+1e-16)
    tmp = tf.reduce_sum(rrr * (-tf.log(rrr+1e-16)),3)
    return(tmp)
