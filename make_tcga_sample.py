import os
import sys
import csv
import numpy as np
import pickle
import myutil
from PIL import Image

random_seed = 765
np.random.seed(random_seed)
print('random_seed',random_seed)

ns = 300       # number of sample files 

nx = ny = 32   # image size 
nn = 100       # number of slices from a image

nx = ny = 64   # image size 
nn = 100       # number of slices from a image

nx = ny = 128  # image size 
nn = 100        # number of slices from a image

nl = 3         # RGB

print("image size",nx,ny)
print("number of train images",nn*ns)

dir_tcga = None
dir_tcga_project   = '/project/hikaku_db/data/tissue_images'
dir_tcga_local_mac = '/Users/nono/Documents/data/tissue_images'
if(os.path.exists(dir_tcga_project)):
    dir_tcga = dir_tcga_project
if(os.path.exists(dir_tcga_local_mac)):
    dir_tcga = dir_tcga_local_mac
if(dir_tcga == None):
    print('cannot find data dir')

#list_dir_src = ("TCGA-05-4384-01A-01-BS1_files/15/",
#                "TCGA-38-4631-11A-01-BS1_files/15/",
#                "TCGA-05-4425-01A-01-BS1_files/15/")
# ns = len(list_dir_src) # number of source images
# file_imglist = 'filelist.txt'

dir_input_Users = '/Users/nono/Documents/data/tissue_images'
dir_input_home = '/home/nono/Documents/data/tissue_images'
if os.path.exists(dir_input_home) :
    dir_input = os.path.join(dir_input_home,'input_w{}'.format(nx))
if os.path.exists(dir_input_Users) :
    dir_input = os.path.join(dir_input_Users,'input_w{}'.format(nx))

if(not os.path.exists(dir_input)):
    os.makedirs(dir_input)

print(dir_input)
dir_image = dir_tcga_project
dir_data = 'dat1'

print(dir_data)

file_imglist = 'typelist.filterd.txt'
fileTable = list(csv.reader(open("typelist.filterd.txt",'r'), delimiter='\t'))


iii_sample = np.random.choice(range(len(fileTable)),size=ns,replace=False)

qqq_src = []
for aa in range(ns):
    ii = iii_sample[aa]
    
    file_src = fileTable[ii][0]
    print(file_src)
    img_src = Image.open(os.path.join(dir_image,file_src),'r')
    mx = img_src.size[0]
    my = img_src.size[1]

    qqq_ss = np.empty((nn, nx, ny, nl), np.float32)
    bb = 0
    while(bb < nn):
        x0 = np.random.choice(range(mx-nx),size=1)[0]
        y0 = np.random.choice(range(my-ny),size=1)[0]

        img_tmp = img_src.crop((x0,y0,x0+nx,y0+ny))
        qqq_tmp = np.asarray(img_tmp) / 255.0
        sd_tmp = np.std(np.asarray(img_tmp))
        if sd_tmp < 16 :
            continue # skip (almost) blank slice
        qqq_ss[bb,] = qqq_tmp
        bb += 1
        ## print(bb)
    qqq_src.append(qqq_ss)

qqq_src = np.vstack(qqq_src)
np.save(os.path.join(dir_data,'tcga_trn_w{}.npy'.format(nx)), qqq_src)
