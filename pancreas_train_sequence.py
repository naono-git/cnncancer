#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

print('train stacked autoencoder stage 1')

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

trainable1 = True
trainable2 = True
trainable3 = True
trainable4 = True
tmax = 10
tprint = 1

if(False):
    exec(open('pancreas_make_sample.py').read())

if(stamp1=='NA'):
    exec(open('pancreas_train_stage1.py').read())
    stamp1 = stamp
    trainable1 = False
    exec(open('pancreas_make_encode1.py').read())

if(stamp2=='NA'):
    exec(open('pancreas_train_stage2.enc.py').read())
    stamp2 = stamp
    trainable2 = False
    exec(open('pancreas_make_encode2.py').read())

if(stamp3=='NA'):
    exec(open('pancreas_train_predict_ki2.py').read())
