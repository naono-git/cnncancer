#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('setup stacked autoencoder stage3')

#
# setup parameters
#

# extern stamp3
if(stamp3=='NA'):
    print('initialize w3 and b3 randomly')
    weight3 = {
        'conv3':   tf.Variable(tf.truncated_normal([fs_3,fs_3,nf_encode2,nf_conv3],stddev=0.05)),
        'encode3': tf.Variable(tf.truncated_normal([fs_3,fs_3,nf_conv3,nf_encode3],stddev=0.05)),
        'hidden3': tf.Variable(tf.truncated_normal([fs_3,fs_3,nf_encode3,nf_conv3],stddev=0.05)),
        'deconv3': tf.Variable(tf.truncated_normal([fs_3,fs_3,nf_conv3,nf_encode2],stddev=0.05)),
    }
    bias3 = {
        'conv3':   tf.Variable(tf.zeros([nf_conv3])),
        'encode3': tf.Variable(tf.zeros([nf_encode3])),
        'hidden3': tf.Variable(tf.zeros([nf_conv3])),
        'deconv3': tf.Variable(tf.zeros([nf_encode2])),
    }
else:
    print('load w3 and b3 from',stamp3)
    file_w3 = 'weight3.{}.pkl'.format(stamp3)
    file_b3 = 'bias3.{}.pkl'.format(stamp3)
    
    path_w3 = os.path.join('out1',file_w3)
    path_b3 = os.path.join('out1',file_b3)

    if(not os.path.exists(path_w3) or not os.path.exists(path_b3)):
        myutil.getRemoteFile([file_w3,file_b3])

    weight3 = tensorflow_ae_base.load_tf_variable(path_w3,key3,trainable=trainable3)
    bias3   = tensorflow_ae_base.load_tf_variable(path_b3,key3,trainable=trainable3)
#end if(stamp=='NA')

#
# setup layers
#

def get_encode3(qqq):
    conv3   = tf.nn.relu(conv2d(qqq,    weight3['conv3'],  bias3['conv3']))
    pool3   = max_pool(conv3,kk=pool_size)
    encode3 = tf.nn.relu(conv2d(pool3,  weight3['encode3'],bias3['encode3']))
    return(encode3)

def get_deconv3(qqq):
    hidden3 = tf.nn.relu(conv2d(qqq,    weight3['hidden3'],bias3['hidden3']))
    unpool3 = un_pool(hidden3,kk=pool_size)
    deconv3 = tf.nn.relu(conv2d(unpool3,weight3['deconv3'],bias3['deconv3']))
    return(deconv3)

#
# entropy function
#

def get_local_entropy_encode3(qqq):
    ooo = tf.reduce_sum(qqq,3,keep_dims=True)
    rrr = qqq / (tf.tile(ooo,[1,1,1,nf_encode3])+1e-16)
    tmp = tf.reduce_sum(rrr * (-tf.log(rrr+1e-16)),3)
    return(tmp)

#
# save network parameters
#
def save_stage3():
    weight3_fin = {k:sess.run(v) for k,v in weight3.items()}
    bias3_fin = {k:sess.run(v) for k,v, in bias3.items()}
    myutil.saveObject(weight3_fin,'weight3.{}.pkl'.format(stamp))
    myutil.saveObject(bias3_fin,'bias3.{}.pkl'.format(stamp))
    return([weight3_fin,bias3_fin])


