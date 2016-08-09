#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('setup classyfiler stage 4')

nf_conv4 = 24
nf_encode4 = 3
key4 = ['conv','encode']

if(stamp4=='NA'):
    print('initialize w4 and b4 randomly')
    weight4 = {
        'conv':   tf.Variable(tf.truncated_normal([16,16,12,24],stddev=0.1)),
        'encode': tf.Variable(tf.truncated_normal([16,16,24,3],stddev=0.1)+1e-3),
    }
    bias4 = {
        'conv':   tf.Variable(tf.zeros([nf_conv4])+0.1),
        'encode': tf.Variable(tf.zeros([nf_encode4])+10),
    }
else:
    print('load w4 and b4 from',stamp4)
    file_w4 = 'weight4.{}.pkl'.format(stamp4)
    file_b4 = 'bias4.{}.pkl'.format(stamp4)
    
    path_w4 = os.path.join('out1',file_w4)
    path_b4 = os.path.join('out1',file_b4)

    if(not os.path.exists(path_w4) or not os.path.exists(path_b4)):
        myutil.getRemoteFile([file_w4,file_b4])

    weight4 = tensorflow_ae_base.load_tf_variable(path_w4,key4,trainable=trainable4)
    bias4   = tensorflow_ae_base.load_tf_variable(path_b4,key4,trainable=trainable4)
#end if(stamp=='NA')

#
# setup layers
#

def get_encode4(qqq):
    qqq1 = tf.nn.conv2d(qqq, weight4['conv'], strides=[1, 16, 16, 1], padding='VALID')
    qqq2 = tf.nn.bias_add(qqq1,bias4['conv'])
    qqq3 = tf.nn.relu(qqq2)
    qqq4 = tf.nn.conv2d(qqq3, weight4['encode'], strides=[1, 1, 1, 1], padding='VALID')
    qqq5 = tf.nn.bias_add(qqq4,bias4['encode'])
    qqq6 = tf.nn.relu(qqq5)
    return(qqq6)

#
# score function
#
def loss_class4(src,dst):
    loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels, name=None)
    return(loss)

def get_local_entropy_encode4(qqq):
    ooo = tf.reduce_sum(qqq,4,keep_dims=True)
    rrr = qqq / (tf.tile(ooo,[1,1,1,nf_encode4])+1e-16)
    tmp = tf.reduce_sum(rrr * (-tf.log(rrr+1e-16)),4)
    return(tmp)

def verify_class(xxx,yyy):
    nn = xxx.shape[0]
    if(nn != len(yyy)):
        print("xxx.shape[0]:",xxx.shape[0],"!=len(yyy):",len(yyy),"\n")
        return(0)
    iii_bin = np.arange(batch_size,nn,batch_size)
    iii_nn = np.arange(nn)
    iii_batches = np.split(iii_nn,iii_bin)

    tmpx = []
    tmpy = []
    for iii in iii_batches:
        tmp = tf_encode4.eval({tf_encode2:xxx[iii,],tf_yyy:yyy[iii]})
        tmpx.append(np.argmax(tmp[:,0,0,:],axis=1))
        tmpy.append(yyy[iii])
    hoge = np.hstack(tmpx)
    fuga = np.hstack(tmpy)
    return(np.transpose(np.vstack((hoge,fuga))))

#
# save network parameters
#
def save_stage4():
    weight4_fin = {k:sess.run(v) for k,v in weight4.items()}
    bias4_fin = {k:sess.run(v) for k,v, in bias4.items()}
    myutil.saveObject(weight4_fin,'weight4.{}.pkl'.format(stamp))
    myutil.saveObject(bias4_fin,'bias4.{}.pkl'.format(stamp))
    return([weight4_fin,bias4_fin])
