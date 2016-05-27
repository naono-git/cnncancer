#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('setup stacked autoencoder stage2')

#
# setup parameters
#

# extern stamp2
if(stamp2=='NA'):
    print('initialize w2 and b2 randomly')
    weight2 = {
        'conv2':   tf.Variable(tf.truncated_normal([fs_2,fs_2,nf_encode1,nf_conv2],stddev=0.05)),
        'encode2': tf.Variable(tf.truncated_normal([fs_2,fs_2,nf_conv2,nf_encode2],stddev=0.05)),
        'hidden2': tf.Variable(tf.truncated_normal([fs_2,fs_2,nf_encode2,nf_conv2],stddev=0.05)),
        'deconv2': tf.Variable(tf.truncated_normal([fs_2,fs_2,nf_conv2,nf_encode1],stddev=0.05)),
    }
    bias2 = {
        'conv2':   tf.Variable(tf.zeros([nf_conv2])),
        'encode2': tf.Variable(tf.zeros([nf_encode2])),
        'hidden2': tf.Variable(tf.zeros([nf_conv2])),
        'deconv2': tf.Variable(tf.zeros([nf_encode1])),
    }
else:
    print('load w2 and b2 from',stamp2)
    file_w2 = 'weight2.{}.pkl'.format(stamp2)
    file_b2 = 'bias2.{}.pkl'.format(stamp2)

    path_w2 = os.path.join('out1',file_w2)
    path_b2 = os.path.join('out1',file_b2)

    if(not os.path.exists(path_w2) or not os.path.exists(path_b2)):
        myutil.getRemoteFile([file_w2,file_b2])

    weight2 = tensorflow_ae_base.load_tf_variable(path_w2,key2,trainable=trainable2)
    bias2   = tensorflow_ae_base.load_tf_variable(path_b2,key2,trainable=trainable2)
#end if(stamp=='NA')

#
# setup layers
#

def get_encode2(qqq):
    conv2   = tf.nn.relu(conv2d(qqq,    weight2['conv2'],  bias2['conv2']))
    pool2   = max_pool(conv2,kk=pool_size)
    encode2 = tf.nn.relu(conv2d(pool2,  weight2['encode2'],bias2['encode2']))
    return(encode2)

def get_deconv2(qqq):
    hidden2 = tf.nn.relu(conv2d(qqq,    weight2['hidden2'],bias2['hidden2']))
    unpool2 = un_pool(hidden2,kk=pool_size)
    deconv2 = tf.nn.relu(conv2d(unpool2,weight2['deconv2'],bias2['deconv2']))
    return(deconv2)

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
def save_stage2():
    weight2_fin = {k:sess.run(v) for k,v in weight2.items()}
    bias2_fin = {k:sess.run(v) for k,v, in bias2.items()}
    myutil.saveObject(weight2_fin,'weight2.{}.pkl'.format(stamp))
    myutil.saveObject(bias2_fin,'bias2.{}.pkl'.format(stamp))
    return([weight2_fin,bias2_fin])


