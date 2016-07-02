#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('setup stacked autoencoder stage2')

#
# setup parameters
#

#
# save and load
#
def save_stage2():
    weight2_fin = {k:sess.run(v) for k,v in weight2.items()}
    myutil.saveObject(weight2_fin,'weight2.{}.pkl'.format(stamp))
    return(weight2_fin)

def load_stage2(path):
    weight2  = tensorflow_ae_base.load_tf_variable(path,key1,trainable=trainable1)
    return(weight2)

# extern stamp2
if(stamp2=='NA'):
    print('initialize w2 and b2 randomly')
    weight2 = {
        'conva':  tf.Variable(tf.truncated_normal([fs_2,fs_2,nf_encode1,nf_conv2],stddev=0.05)),
        'convb':  tf.Variable(tf.truncated_normal([fs_2,fs_2,nf_encode1,nf_conv2],stddev=0.05)),
        'encode': tf.Variable(tf.truncated_normal([fs_2,fs_2,nf_conv2*2,nf_encode2],stddev=0.05)),
    }
else:
    print('load weight2 from',stamp2)
    file_w2 = 'weight2.{}.pkl'.format(stamp2)
    path_w2 = os.path.join('out1',file_w2)
    if(not os.path.exists(path_w2)):
        myutil.getRemoteFile(file_w2)

    weight2 = load_stage2(path_w2)
#end if(stamp=='NA')

#
# setup layers
#

def get_encode2(qqq):
    conv2a  = tf.nn.conv2d(qqq,weight2['conva'],strides=[1,1,1,1],padding='SAME')
    pool2a  = max_pool(conv2a,kk=2)
    conv2b  = tf.nn.conv2d(qqq,weight2['convb'],strides=[1,2,2,1],padding='SAME')
    concat2 = tf.concat(3,(pool2a,conv2b))
    encode2 = tf.nn.conv2d(concat2,weight2['encode'],strides=[1,1,1,1],padding='SAME')
    relu2   = tf.nn.relu(encode2)
    return(relu2)

def get_deconv2(qqq):
    decode2  = tf.nn.conv2d_transpose(qqq,weight2['encode'],output_shape=[batch_size,ny//4,nx//4,nf_conv2*2],strides=[1,1,1,1],padding='SAME')
    unpool2a = un_pool_2(tf.slice(decode2,[0,0,0,0],[batch_size,ny//4,nx//4,nf_conv2]))
    deconv2a = tf.nn.conv2d_transpose(unpool2a,weight2['conva'],output_shape=[batch_size,ny//2,nx//2,nf_encode1],strides=[1,1,1,1],padding='SAME')
    deconv2b = tf.nn.conv2d_transpose(tf.slice(decode2,[0,0,0,nf_conv2],[batch_size,ny//4,nx//4,nf_encode1]),weight2['convb'],output_shape=[batch_size,ny//2,nx//2,nf_encode1],strides=[1,2,2,1],padding='SAME')
    concat2  = tf.dynamic_stitch([[0,2,4,6,8,10],[1,3,5,7,9,11]], (tf.transpose(deconv2a),tf.transpose(deconv2b)))
    merge2   = tf.transpose(tf.segment_mean(concat2,[0,0,1,1,2,2,3,3,4,4,5,5]))
    relu2    = tf.nn.relu(merge2)
    return(relu2)

#
# entropy function
#

def get_local_entropy_encode2(qqq):
    ooo = tf.reduce_sum(qqq,3,keep_dims=True)
    rrr = qqq / (tf.tile(ooo,[1,1,1,nf_encode2])+1e-16)
    tmp = tf.reduce_sum(rrr * (-tf.log(rrr+1e-16)),3)
    return(tmp)

#
# save network parameters
#
