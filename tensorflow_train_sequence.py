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

ss = 32
# exec(open('make_tcga_sample.py').read())
# del(qqq_trn)
exec(open('tensorflow_train_stage1.py').read())
stamp1 = stamp
trainable1 = False

exec(open('make_tcga_encoded1.py').read())

exec(open('tensorflow_train_stage2.enc.py').read())
stamp2 = stamp
trainable2 = False

exec(open('make_tcga_encoded2.py').read())

exec(open('tensorflow_train_stage3.enc.py').read())
stamp3 = stamp
trainable3 = False

exec(open('make_tcga_encoded2_2048.py').read())
del(qqq_trn)

exec(open('tensorflow_train_stage4_encoded2.py').read())
stamp4 = stamp
