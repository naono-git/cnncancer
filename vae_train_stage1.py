
print('train variational autoencoder stage 1')

import os
import sys
import csv
import numpy as np
import numpy as np
import pickle
from PIL import Image

import tensorflow as tf
import sonnet as snt

## time stamp
stamp = myutil.show_timestamp()
print('stamp = ',stamp)

dir_data = configparser.get('train', 'dir_data')[1:-1]
dir_out = configparser.get('train', 'dir_out')[1:-1]

## setup parameters
file_par = 'par1/par1.txt'
configparser = configparser.ConfigParser()
configparser.read(file_pat)

if(random_seed < 0):
    random_seed = int(stamp1) % 4294967295

np.random.seed(random_seed)

varname = 'nt_max'
if(not varname in globals()):
    eq = '{} = {}'.format(varname,v0)
    print(eq)
    exec(eq,globals(),locals())
    eq = '{} = {}'.format(varname,v0)

nt_max = configparser.getint('train', 'nt_max')

if(not 'sess' in locals()):
    print('create a new interactive session')
    sess = tf.InteractiveSession()
else:
    tf.reset_default_graph()


## load data

nx = 32 # sample size
file_trn = 'pancreas_he_w{}.npy'.format(nx)
path_trn = os.path.join(dir_data, file_trn)
print('load input from {}'.format(path_trn))
qqq_trn = np.load(path_trn)

nn,ny,nx,nl = qqq_trn.shape
print('nn ny nx nl',nn,ny,nx,nl)
