#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('verify stage2 predictor ki67')

import os
import sys
import csv
import numpy as np
import numpy as np
import pickle
from PIL import Image

import tensorflow as tf
import tensorflow_ae_base

from tensorflow_ae_base import *
import tensorflow_util

import myutil

exec(open('extern_params.py').read())

#
# load sample data
#
dir_data = 'dat1'

list_img_file_he = ('KPC F838-2/HE/KPC-F838-2_2015_08_26_0002.tif',6732, 1332)

list_img_file_ki = ('KPC F838-2/Ki67/KPC-F838-2_2015_10_05_0010.tif',6748, 1148)

xs,ys = 1024,1024

path_src = os.path.join(dir_data,'img0_1K.tif')
img_src_1K = Image.open(path_src,'r')

path_dst = os.path.join(dir_data,'img1_1K.tif')

if(True):
    img1_1K = Image.open(path_dst,'r')
else:
    ff_dst = list_img_file_ki[0]
    x1,y1 = list_img_file_ki[1],list_img_file_ki[2]
    path_dst = os.path.join(dir_img,ff_dst)
    img1 = Image.open(path_dst,'r')
    img1_1K = img1.crop((x1,y1,x1+xs,y1+ys))

qqq_vry_src = np.asarray(img0_1K)[np.newaxis] / 255.0
qqq_vry_dst = np.asarray(img1_1K)[np.newaxis] / 255.0

nn,ny,nx,nl = qqq_vry_src.shape
nn = 1
print('ny nx nl',ny,nx,nl)

exec(open('tensorflow_ae_stage1.py').read())
exec(open('tensorflow_ae_stage2.py').read())
exec(open('pancreas_layer_predict3_ki.py').read())

tf_src = tf.placeholder(tf.float32, [None,ny,nx,nl])
tf_dst = tf.placeholder(tf.float32, [None,ny,nx,nl])
tf_dst3 = get_dist3(tf_dst)
tf_encode1 = get_encode1(tf_src)
tf_deconv1 = get_deconv1(tf_encode1)
tf_encode2 = get_encode2(tf_encode1)
tf_deconv2 = get_deconv2(tf_encode2)
tf_deconv1b = get_deconv1(tf_deconv2)
tf_predict3 = get_predict3(tf_encode2)
mean_error = tf.reduce_mean(tf.square(tf_predict3 - tf_dst3))

sess.run(tf.global_variables_initializer())

mean_error_out = mean_error.eval({tf_src: qqq_src,tf_dst: qqq_dst})
print(mean_error_out)

hoge = tf_encode1.eval({tf_src: qqq_vry_src})
fuga = tensorflow_util.get_image_from_encode(hoge)
# fuga.show()

qqq_dst3 = tf_dst3.eval({tf_dst: qqq_vry_dst})
img_dst3 = tensorflow_util.get_image_from_qqq(qqq_dst3)
img_dst3.save('out1/dst3.jpg'.format(stamp3))

np.savetxt('out1/qqq_dst.txt',qqq_dst3.reshape((256,256*3)))
np.savetxt('out1/qqq_predict3.txt',qqq_predict3.reshape((256,256*3)))

qqq_predict3 = tf_predict3.eval({tf_src: qqq_vry_src})
# img_out = tensorflow_util.get_image_from_encode(qqq_predict)
img_out = tensorflow_util.get_image_from_qqq(qqq_predict3)
# img_out.show()
img_out.save('out1/predict3.{}.jpg'.format(stamp3))

qqq_src3 = tf_src3.eval({tf_src: qqq_vry_src})
img_src3 = tensorflow_util.get_image_from_qqq(qqq_src3)
img_cmp = myutil.cbind_image(myutil.cbind_image(img_src3,img_out),img_dst3)
img_cmp.save('out1/cmp3.{}.jpg'.format(stamp3))

img_w1 = tensorflow_util.get_image_from_ww(weight1['deconv1'].eval()[:,:,:,0])
img_w1 = myutil.cbind_image(img_w1,tensorflow_util.get_image_from_ww(weight1['deconv1'].eval()[:,:,:,1]))
img_w1 = myutil.cbind_image(img_w1,tensorflow_util.get_image_from_ww(weight1['deconv1'].eval()[:,:,:,2]))
img_w1 = myutil.cbind_image(img_w1,tensorflow_util.get_image_from_ww(weight1['deconv1'].eval()[:,:,:,3]))
img_w1 = myutil.cbind_image(img_w1,tensorflow_util.get_image_from_ww(weight1['deconv1'].eval()[:,:,:,4]))
img_w1 = myutil.cbind_image(img_w1,tensorflow_util.get_image_from_ww(weight1['deconv1'].eval()[:,:,:,5]))
img_w1.save('out1/img_w1.jpg')

qqq_encode2 = tf_encode2.eval({tf_src: qqq_vry_src})
np.savetxt('out1/qqq_encode2.txt',qqq_encode2.reshape((256,256*12)))

if(True):
    qqq_encode1 = tf_encode1.eval({tf_src: qqq_src})
    img_encode1 = tensorflow_util.get_image_from_encode(qqq_encode1)
    img_encode1.show()

    qqq_deconv1 = tf_deconv1.eval({tf_src: qqq_src})
    img_deconv1 = tensorflow_util.get_image_from_qqq(qqq_deconv1)
    img_deconv1.show()

if(True):
    qqq_encode2 = tf_encode2.eval({tf_src: qqq_src})
    img_encode2 = tensorflow_util.get_image_from_encode(qqq_encode2)
    # img_encode2.show()

    qqq_deconv2 = tf_deconv2.eval({tf_src: qqq_src})
    img_deconv2 = tensorflow_util.get_image_from_encode(qqq_deconv2)
    # img_deconv2.show()

    qqq_deconv1b = tf_deconv1b.eval({tf_src: qqq_src})
    img_deconv1b = tensorflow_util.get_image_from_qqq(qqq_deconv1b)
    # img_deconv1b.show()

import rpy2
import rpy2.robjects
import rpy2.robjects.numpy2ri

if(not stamp3 == 'NA'):
    qqq_predict3 = tf_predict3.eval({tf_src: qqq_src})
    img_predict3 = tensorflow_util.get_image_from_qqq(qqq_predict3)
    img_predict3.show()

    img_src = img_src_1K
    qqq_dst = tf_dst3.eval({tf_dst: qqq_dst})
    img_dst = tensorflow_util.get_image_from_qqq(qqq_dst)
    img_dst.show()

    if(False):
        r_qqq_dst = rpy2.robjects.numpy2ri.numpy2ri(qqq_dst)
        r_qqq_predict3 = rpy2.robjects.numpy2ri.numpy2ri(qqq_predict3)

        rpy2.robjects.r.assign('qqq_dst',r_qqq_dst)
        rpy2.robjects.r.assign('qqq_predict3',r_qqq_predict3)

        rpy2.robjects.r("save(qqq_dst,file='qqq_dst.gzip')")
        rpy2.robjects.r("save(qqq_predict3,file='qqq_predict3.gzip')")
