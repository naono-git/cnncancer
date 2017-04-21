#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

import myutil

def save_params(dirname='.'):
    path_params = os.path.join(dirname, 'params.{}.pkl'.format(stamp))
    pickle.dump((extern_params,network_params),open(path_params,'wb'))
    print(path_params)
    return(path_params)

def save_params_txt(dirname='.'):
    path_params = os.path.join(dirname, 'params.{}.txt'.format(stamp))
    with open(path_params, "w") as fout:
        for k,v in extern_params.items():
            if(type(v)==str):
                print('{} = \'{}\''.format(k,v),file=fout)
            else:
                print('{} = {}'.format(k,v),file=fout)
        for k,v in network_params.items():
            if(type(v)==str):
                print('{} = \'{}\''.format(k,v),file=fout)
            else:
                print('{} = {}'.format(k,v),file=fout)

extern_params_table = """\
random_seed = 'NA'
stamp1 = 'NA'
stamp2 = 'NA'
stamp3 = 'NA'
trainable1 = 'NA'
trainable2 = 'NA'
trainable3 = 'NA'
tmax   = 3
tprint = 1
learning_rate = 1e-3
keep_prob_1   = 1.0
batch_size    = 32
\
"""
extern_params = {'random_seed'  : 'NA',
                 'stamp1'       : 'NA',
                 'stamp2'       : 'NA',
                 'stamp3'       : 'NA',
                 'trainable1'   : True,
                 'trainable2'   : True,
                 'trainable3'   : True,
                 # time steps
                 'tmax'      : 3,
                 'tprint'    : 1,
                 # learning parameters
                 'learning_rate' : 1e-3,
                 'lambda_risa'   : 0,
                 'keep_prob_1'   : 1.0,
                 'batch_size'    : 32}
#
# set default values if they are not defined yet
#
for k,v in extern_params.items():
    if(not k in globals()):
        if(type(v)==str):
            print('{} = \'{}\''.format(k,v))
            exec( '{} = \'{}\''.format(k,v),globals(),locals())
        else:
            print('{} = {}'.format(k,v))
            exec('{} = {}'.format(k,v),globals(),locals())
    else:
        extern_params[k] = globals()[k]
#

#       
# current time stamp
#
stamp = myutil.show_timestamp()
print('stamp = ',stamp)

#
# random seed
#
if(random_seed=='NA'):
    rs = int(stamp) % 4294967295
    print('random_seed = NA, use tmp seed = ',rs)
    np.random.seed(rs)
    random_seed = rs
else:
    print('random_seed = ',random_seed)
    np.random.seed(random_seed)

#
# data paths
#

# it may depends on machines...
username = os.environ['USER']
dir_project = '/project/hikaku_db/data/tissue_images/'
dir_home    = '/home/{}/Documents/data/tissue_images/'.format(username)
dir_Users   = '/Users/{}/Documents/data/tissue_images/'.format(username)

if(not 'dir_image' in locals()):
    dir_image = 'NA'
if(os.path.exists(dir_project)):
    dir_image = dir_project
else :
    if(os.path.exists(dir_Users)):
        dir_image = dir_Users

if(not 'dir_input' in locals()):
    dir_input = 'NA'
if(os.path.exists(dir_home)):
    dir_input = dir_home
if(os.path.exists(dir_Users)):
    dir_input = dir_Users

if(not 'dir_data' in locals()):
    dir_data = 'dat1'
if(not 'dir_out' in locals()):
    dir_out = 'out1'
#

#
# network structures
#

network_params = {
    # number of  filters
    'nf_RGB'     : 3,
    'nf_conv1'   : 3,
    'nf_encode1' : 3,
    'nf_conv2'   : 6,
    'nf_encode2' : 6,
    'nf_conv3'   : 12,
    'nf_encode3' : 12,
    # filter size and pad size
    'fs_1' : 7,
    'fs_2' : 5,
    'fs_3' : 3,
    'pool_size' : 2}
#

# always overwrite
for k,v in network_params.items():
    if(type(v)==str):
        exec( '{} = \'{}\''.format(k,v),globals(),locals())
    else:
        exec('{} = {}'.format(k,v),globals(),locals())
#

key2 = ['conv', 'encode', 'hidden', 'deconv']
key3 = ['encode', 'decode']

#
# setup tensorflow session
#

if(not 'sess' in locals()):
    print('create a new interactive session')
    sess = tf.InteractiveSession()
#
