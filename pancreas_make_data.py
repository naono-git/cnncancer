import os
import sys
import numpy as np
from PIL import Image

random_seed = 765
np.random.seed(random_seed)
print('random_seed',random_seed)

nl = 3
#nx = ny = 32   # image size
#nn = 10000     # number of slices for each image

# nx = ny = 64   # image size                                                                         
# nn = 2560      # number of slices for each image                                                    

nx = ny = 128  # image size                                                                         
nn = 1280      # number of slices for each image                                                    
print(nx)

### NOTE chage these paths according to your environment

if(os.path.exists('/Users/nono/Documents/data/KPC')):
    dir_img = '/Users/nono/Documents/data/KPC'
if(os.path.exists('/project/hikaku_db/data/KPC')):
    dir_img = '/project/hikaku_db/data/KPC'

list_img_file = ('KPC F838-2/HE/KPC-F838-2_2015_08_26_0002.tif',
                 'KPC F838-3/HE/KPC-F838-3_2016_07_12_014.tif',
                 'KPC K-135/K135 Panc HE_s1.tif',
                 'KPC K-138/K138 Panc HE_s1.tif',
                 'KPC K-97/K97 Panc HE_s1.tif',
                 'KPC K-99/K99 Panc HE_s1.tif',
                 'KPCL103/KPCL103 Panc HE.tif')

list_img_file = ('KPC F838-2/HE/KPC-F838-2_2015_08_26_0002.tif',
                 'KPC F838-3/HE/KPC-F838-3_2016_07_12_014.tif') # add more
dir_data = 'dat1'

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
np.save(os.path.join(dir_data,'pancreas_w{}.npy').format(nx),data_all)
