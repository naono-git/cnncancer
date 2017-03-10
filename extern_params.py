#!/usr/bin/env python
# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

import myutil

extern_params = {'random_seed'  : 'NA',
                 'stamp1'       : 'NA',
                 'stamp2'       : 'NA',
                 'stamp3'       : 'NA',
                 'stamp4'       : 'NA',
                 'trainable1'   : True,
                 'trainable2'   : True,
                 'trainable3'   : True,
                 'trainable4'   : True,
                 'index_filter1': None,
                 'index_filter2': None,
                 'index_filter3': None,
                 # time steps
                 'tmax'      : 3,
                 'tprint'    : 1,
                 # remote server
                 'dirname_cur'   : 'Documents/cnncancer_pancreas/out1',
                 'hostname_cur' : 'csb85',
                 'portnum_cur'  : None}
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
    'nf_conv1'   : 6,
    'nf_encode1' : 6,
    'nf_conv2'   : 9,
    'nf_encode2' : 9,
    'nf_conv3'   : 12,
    'nf_encode3' : 12,
    'nf_conv4'   : 24,
    'nf_class4'  : 3,
    # filter size and pad size
    'fs_1' : 7,
    'ps_1' : 3,
    'fs_2' : 5,
    'ps_2' : 2,
    'fs_3' : 3,
    'ps_3' : 1,
    'fs_4' : 16,
    'pool_size' : 2,
    # learning parameters
    'lambda_s'      : 1e-5,
    'learning_rate' : 0.01,
    'batch_size'    : 32}
#

# always overwrite
for k,v in network_params.items():
    if(type(v)==str):
        print('{} = \'{}\''.format(k,v))
        exec( '{} = \'{}\''.format(k,v),globals(),locals())
    else:
        print('{} = {}'.format(k,v))
        exec('{} = {}'.format(k,v),globals(),locals())
#

key1 = ['conv1', 'encode1', 'hidden1', 'deconv1']
key2 = ['conv2', 'encode2', 'hidden2', 'deconv2']
key3 = ['encode', 'decode']

#
# setup tensorflow session
#

if(not 'sess' in locals()):
    print('create a new interactive session')
    sess = tf.InteractiveSession()
#

def get_params():
    params = dict()
    for kk in extern_params.keys():
        params[kk] = globals()[kk]
    for kk in network_params.keys():
        params[kk] = globals()[kk]
    return(params)
#
