import os
import sys
import numpy as np
from PIL import Image

random_seed = 123456
np.random.seed(random_seed)
print('random_seed',random_seed)

nl = 3

nx = ny = 32   # image size
## nn = 20480     # number of slices for each image
nn = 100     # number of slices for each image

#nx = ny = 64   # image size
#nn = 2560      # number of slices for each image

#nx = ny = 128  # image size
#nn = 1280      # number of slices for each image

print(nx)

### NOTE chage these paths according to your environment

if(os.path.exists('/Users/nono/Documents/data/KPC')):
    dir_img = '/Users/nono/Documents/data/KPC'
if(os.path.exists('/project/hikaku_db/data/KPC')):
    dir_img = '/project/hikaku_db/data/KPC'

list_img_file = ('KPC F838-2/Ki67/KPC-F838-2_2015_10_05_0010.tif',
                 'KPC F838-3/Ki67/KPC-F838-3_2016_09_15_027.tif')

dir_data = 'dat1'

###

data_list = []
data_set = np.empty((nn,nx,ny,nl),np.float32)

for aa in range(2):
## for ff in list_img_file:
    ff = list_img_file[aa]
    print(ff)
    path_img = os.path.join(dir_img,ff)
    img_src = Image.open(path_img,'r')
    mx,my = img_src.size
    ii = 0
    file_tmp = open('tmp.{}.txt'.format(aa),'w')
    while ii < nn:
        x0 = np.random.choice(range(mx-nx),size=1)[0]
        y0 = np.random.choice(range(my-ny),size=1)[0]
        img_tmp = img_src.crop((x0,y0,x0+nx,y0+ny))
        qqq_tmp = np.asarray(img_tmp) / 256.0
        sd_tmp = np.std(np.asarray(img_tmp))
        if sd_tmp < 60 :
            continue # skip (almost) blank subframe
        hoge = np.mean(qqq_tmp,axis=(0,1))
        print("{}\t{}\t{}\t{}".format(hoge[0],hoge[1],hoge[2],sd_tmp),file=file_tmp)
        data_set[ii,] = qqq_tmp
        ii += 1
    data_list.append(data_set)

data_all = np.vstack(data_list)
print(data_all.shape)
np.save(os.path.join(dir_data,'pancreas_ki_w{}_out.npy').format(nx),data_all)
