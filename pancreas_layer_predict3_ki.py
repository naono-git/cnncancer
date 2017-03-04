#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('setup stacked pancreas predictor stage2')
exec(open('extern_params.py').read())

#
# setup parameters
#

# extern stamp2
if(stamp2=='NA'):
    print('initialize w2 and b2 randomly')
    weight2 = {
        'encode': tf.Variable(tf.truncated_normal([fs_2,fs_2,nf_conv2,nf_encode2],stddev=0.05)),
        'decode': tf.Variable(tf.truncated_normal([fs_2,fs_2,nf_encode2,nf_RGB],stddev=0.05)),
    }
    bias2 = {
        'encode': tf.Variable(tf.zeros([nf_encode2])),
        'decode': tf.Variable(tf.zeros([nf_RGB])),
    }
else:
    print('load w2 and b2 from',stamp2)
    file_w2 = 'weight2.{}.pkl'.format(stamp2)
    file_b2 = 'bias2.{}.pkl'.format(stamp2)

    path_w2 = os.path.join('out1',file_w2)
    path_b2 = os.path.join('out1',file_b2)

    if(not os.path.exists(path_w2) or not os.path.exists(path_b2)):
        myutil.getRemoteFile([file_w2,file_b2],dirname='Documents/cnncancer/out1')

    weight2 = tensorflow_ae_base.load_tf_variable(path_w2,key2,trainable=trainable2)
    bias2   = tensorflow_ae_base.load_tf_variable(path_b2,key2,trainable=trainable2)
#end if(stamp=='NA')

#
# setup layers
#

def get_predict2(qqq):
    encode2 = tf.nn.relu(conv2d(qqq,  weight2['encode'],bias2['encode']))
    predict2 = tf.nn.relu(conv2d(encode2,  weight2['decode'],bias2['decode']))
    return(predict2)

def get_dist2(qqq):
    pool2   = max_pool(qqq,kk=pool_size)
    return(pool2)
#
# save network parameters
#
def save_stage2():
    weight2_fin = {k:sess.run(v) for k,v in weight2.items()}
    bias2_fin = {k:sess.run(v) for k,v, in bias2.items()}
    myutil.saveObject(weight2_fin,'weight2.{}.pkl'.format(stamp))
    myutil.saveObject(bias2_fin,'bias2.{}.pkl'.format(stamp))
    return([weight2_fin,bias2_fin])



