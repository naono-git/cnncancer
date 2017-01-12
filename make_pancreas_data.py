import os
import sys
import numpy as np
from PIL import Image

random_seed = 765
np.random.seed(random_seed)
print('random_seed',random_seed)

nx = ny = 32   # image size
nl = 3
nn = 10000     # number of slices for each image

### NOTE chage these paths according to your environment
dir_img = '/Users/nono/Documents/data/KPC'
list_img_file = ('KPC F838-2/HE/KPC-F838-2_2015_08_26_0002.tif',
                 'KPC F838-3/HE/KPC-F838-3_2016_07_12_014.tif') # add more
dir_data = '/Users/nono/Documents/cnncancer/dat1'
###

data_list = []
data_set = np.empty((nn,nx,ny,nl),np.float32)

for ff in list_img_file:
    print(ff)
    path_img = os.path.join(dir_img,ff)
    img_src = Image.open(path_img,'r')
    mx,my = img_src.size
    ii = 0
    while ii < nn:
        x0 = np.random.choice(range(mx-nx),size=1)
        y0 = np.random.choice(range(my-ny),size=1)
        img_tmp = img_src.crop((x0,y0,x0+nx,y0+ny))
        qqq_tmp = np.asarray(img_tmp) / 255.0
        sd_tmp = np.std(np.asarray(img_tmp))
        if sd_tmp < 16 :
            continue # skip (almost) blank subframe
        data_set[ii,] = qqq_tmp
        ii += 1
    data_list.append(data_set)

data_all = np.vstack(data_list)
print(data_all.shape)
np.save(os.path.join(dir_data,'datafile_w{}.npy').format(nx),data_all)
