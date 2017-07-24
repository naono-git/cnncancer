#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('setup stacked autoencoder stage1')

#
# setup parameters
#

key1 = ['enconv', 'encode', 'decode', 'deconv']

# extern stamp1,trainable1
sd0 = 0.2

weight1 = {
    'enconv': tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_RGB,nf_enconv1],stddev=sd0),trainable=trainable1,name='weight1.enconv'),
    'encode': tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_enconv1,nf_encode1],stddev=sd0),trainable=trainable1,name='weight1.encode'),
    'decode': tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_encode1,nf_enconv1],stddev=sd0),trainable=trainable1,name='weight1.decode'),
    'deconv': tf.Variable(tf.truncated_normal([fs_1,fs_1,nf_enconv1,nf_RGB],stddev=sd0),trainable=trainable1,name='weight1.deconv'),
}
bias1 = {
    'enconv': tf.Variable(tf.zeros([nf_enconv1]),trainable=trainable1, name='bias1.enconv'),
    'encode': tf.Variable(tf.zeros([nf_encode1]),trainable=trainable1, name='bias1.encode'),
    'decode': tf.Variable(tf.zeros([nf_enconv1]),trainable=trainable1, name='bias1.decode'),
    'deconv': tf.Variable(tf.zeros([nf_RGB]    ),trainable=trainable1, name='bias1.deconv'),
}

if(stamp1=='NA'):
    print('initialize w1 and b1 randomly. sd0 = ',sd0)
else:
    print('load w1 and b1 from',stamp1)
    tf_saver.restore(sess,'out1/cnncancer_pancreas_he-{}'.format(stap1))
#     if(stype==''):
#         file_w1 = 'weight1.{}.pkl'.format(stamp1)
#         file_b1 = 'bias1.{}.pkl'.format(stamp1)
#     else:
#         file_w1 = 'weight1.{}.{}.pkl'.format(stype,stamp1)
#         file_b1 = 'bias1.{}.{}.pkl'.format(stype,stamp1)

#     path_w1 = os.path.join(dir_out,file_w1)
#     path_b1 = os.path.join(dir_out,file_b1)

#     if(not os.path.exists(path_w1) or not os.path.exists(path_b1)):
#         myutil.getRemoteFile([file_w1,file_b1],dirname='Documents/cnncancer/out1')
    
#     weight1 = tensorflow_ae_base.load_tf_variable(path_w1,key1,trainable=trainable1)
#     bias1   = tensorflow_ae_base.load_tf_variable(path_b1,key1,trainable=trainable1)
# #end if(stamp=='NA')

#
# setup layers
#

def get_conv1(tf_input):
    tf_1 = tf.nn.conv2d(tf_input, weight1['enconv'], strides=[1, 1, 1, 1], padding='SAME')
    tf_2 = tf.nn.bias_add(tf_1, bias1['enconv'])
    tf_conv = tf.nn.relu(tf_2)
    return(tf_conv)

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


def get_risa1(tf_conv):
    tf_1 = tf.square(tf_conv)
    tf_2 = tf.transpose(tf_1)
    tf_3 = tf.segmentf_sum(tf_2, seg1_3)
    tf_4 = tf.sqrt(tf_3)
    tf_risa = tf.reduce_mean(tf_4)
    return(tf_risa)

def get_encode1(tf_conv):
    tf_pool   = max_pool(tf_conv, kk=pool_size)
    tf_drop   = tf.nn.dropout(tf_pool, keep_prob_1)
    tf_encode = tf.nn.relu(conv2d(tf_drop,  weight1['encode'], bias1['encode']))
    return(tf_encode)

def get_deconv1(tf_encode):
    tf_tmp = tf_encode
    ## tf_tmp = tf.nn.relu(conv2d(tf_encode, weight1['decode'], bias1['decode']))
    ## tf_tmp = un_pool(tf_tmp, kk=pool_size)
    ## tf_tmp = tf.nn.relu(conv2d(tf_tmp, weight1['deconv'], bias1['deconv']))
    tf_tmp = tf.nn.conv2d(tf_tmp, weight1['decode'], 
                         ## outputf_shape=[batch_size, ny//2, nx//2, nf_conv1], 
                         strides=[1, 1, 1, 1], padding='SAME')
    tf_tmp = tf.nn.bias_add(tf_encode, bias1['decode'])
    tf_tmp = tf.nn.relu(tf_tmp)

    tf_tmp = un_pool(tf_tmp, kk=pool_size)

    ##tf_tmp = tf.nn.conv2d_transpose(tf_tmp, weight1['enconv'], 
    ##                               output_shape=[batch_size, ny, nx, nf_RGB],
    ##                               strides=[1, 1, 1, 1], padding='SAME')
    tf_tmp = tf.nn.conv2d(tf_tmp, weight1['deconv'], 
                         strides=[1, 1, 1, 1], padding='SAME')
    tf_tmp = tf.nn.bias_add(tf_tmp, bias1['deconv'])
    tf_tmp = tf.nn.relu(tf_tmp)
    return(tf_tmp)

#
# entropy function
#

def get_local_entropy_encode1(tf_encode):
    tf_ooo = tf.reduce_sum(tf_encode, 3, keep_dims=True)
    tf_rrr = tf_encode / (tf.tile(tf_ooo,[1,1,1,nf_encode1])+1e-16)
    tf_tmp = tf.reduce_sum(tf_rrr * (-tf.log(tf_rrr+1e-16)),3)
    return(tf_tmp)

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


