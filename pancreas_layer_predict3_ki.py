#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('setup stacked pancreas predict3 ki')
exec(open('extern_params.py').read())

#
# setup parameters
#

# extern stamp3
if(stamp3=='NA'):
    print('initialize w3 and b3 randomly')
    weight3 = {
        'encode': tf.Variable(tf.truncated_normal([fs_3,fs_3,nf_encode2,nf_encode3],stddev=0.05)),
        'decode': tf.Variable(tf.truncated_normal([fs_3,fs_3,nf_encode3,nf_RGB],stddev=0.05)),
    }
    bias3 = {
        'encode': tf.Variable(tf.zeros([nf_encode3])),
        'decode': tf.Variable(tf.zeros([nf_RGB])),
    }
else:
    print('load w3 and b3 from',stamp3)
    file_w3 = 'weight3.{}.pkl'.format(stamp3)
    file_b3 = 'bias3.{}.pkl'.format(stamp3)

    path_w3 = os.path.join(dir_out,file_w3)
    path_b3 = os.path.join(dir_out,file_b3)

    if(not os.path.exists(path_w3) or not os.path.exists(path_b3)):
        myutil.getRemoteFile([file_w3,file_b3],dirname_cur,hostname_cur,portnum_cur)

    weight3 = tensorflow_ae_base.load_tf_variable(path_w3,key3,trainable=trainable3)
    bias3   = tensorflow_ae_base.load_tf_variable(path_b3,key3,trainable=trainable3)
#end if(stamp=='NA')

#
# setup layers
#

def get_predict3(qqq):
    encode3 = tf.nn.relu(conv2d(qqq,  weight3['encode'],bias3['encode']))
    predict3 = tf.nn.relu(conv2d(encode3,  weight3['decode'],bias3['decode']))
    return(predict3)

def get_dist3(qqq):
    pool2   = max_pool(qqq,kk=pool_size)
    pool3   = max_pool(pool2,kk=pool_size)
    return(pool3)
#
# save network parameters
#
def save_stage3():
    weight3_fin = {k:sess.run(v) for k,v in weight3.items()}
    bias3_fin = {k:sess.run(v) for k,v, in bias3.items()}
    myutil.saveObject(weight3_fin,'weight3.{}.pkl'.format(stamp))
    myutil.saveObject(bias3_fin,'bias3.{}.pkl'.format(stamp))
    return([weight3_fin,bias3_fin])
