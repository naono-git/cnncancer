#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('setup stacked autoencoder stage1')

#
# setup parameters
#

# merge
# pool 2
# conv1a conv1b(stride2))
# input

# extern stamp1,trainable1
if(stamp1=='NA'):
    print('initialize w1 and b1 randomly')
    weight1 = {
        'conva' : tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_RGB,nf_conv1],stddev=0.05),trainable=trainable1,name='conv1a.ww'),
        'convb' : tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_RGB,nf_conv1],stddev=0.05),trainable=trainable1,name='conv1b.ww'),
        'encode': tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_conv1*2,nf_encode1],stddev=0.05),trainable=trainable1),
        'hidden': tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_encode1,nf_conv1],stddev=0.05),trainable=trainable1),
        'deconv': tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_conv1,nf_RGB],stddev=0.05),trainable=trainable1),
    }
else:
    print('load w1a w1b b1a b1b from',stamp1)
    file_ww1 = 'weight1.{}.pkl'.format(stamp1)

    path_ww1 = os.path.join('out1',file_ww1)

    if(not os.path.exists(path_ww1)):
        myutil.getRemoteFile(file_ww1)
    
    weight1  = tensorflow_ae_base.load_tf_variable(path_w1,key1,trainable=trainable1)
#end if(stamp=='NA')

#
# setup layers
#

def get_encode1(qqq,www):
    conv1a  = tf.nn.conv2d(qqq,www['conva'],strides=[1,1,1,1],padding='SAME')
    pool1a  = max_pool(conv1a,kk=2)
    conv1b  = tf.nn.conv2d(qqq,www['convb'],strides=[1,2,2,1],padding='SAME')
    concat1 = tf.concat(3,(pool1a,conv1b))
    encode1 = tf.nn.conv2d(concat1,www['encode'],strides=[1,1,1,1],padding='SAME')
    relu1   = tf.nn.relu(encode1)
    return(relu1)

def get_deconv1(qqq,www):
    decode1  = tf.nn.conv2d_transpose(qqq,www['encode'],output_shape=[batch_size,ny//2,nx//2,nf_conv1*2],strides=[1,1,1,1],padding='SAME')
    unpool1a = un_pool_2(tf.slice(decode1,[0,0,0,0],[32,16,16,3]))
    deconv1a = tf.nn.conv2d_transpose(unpool1a,www['conva'],output_shape=[32,32,32,3],strides=[1,1,1,1],padding='SAME')
    deconv1b = tf.nn.conv2d_transpose(tf.slice(decode1,[0,0,0,3],[32,16,16,3]),www['convb'],output_shape=[32,32,32,3],strides=[1,2,2,1],padding='SAME')
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

#
# save network parameters
#
def save_stage1():
    www_fin = {k:sess.run(v) for k,v in www.items()}
    bias1_fin = {k:sess.run(v) for k,v, in bias1.items()}
    myutil.saveObject(www_fin,'www.{}.pkl'.format(stamp))
    myutil.saveObject(bias1_fin,'bias1.{}.pkl'.format(stamp))
    return([www_fin,bias1_fin])


