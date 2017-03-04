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

# extern
# nx = ny = 32   # image size 
# sx = sy = 32   # stride size
# na = 1         # number of sample sets
# nn  = 6400 # number of train frames for each set

# nx=ny=sx=sy=32
# na,nn=1,6400

nx=ny=sx=sy=64
na,nn=4,1600

# nx=ny=sx=sy=128
# na,nn=16,400

# nx=ny=sx=sy=256
# na,nn=16,100

## nx=ny=sx=sy=512
## na,nn=4,25

nl = 3

print("image size",nx,ny)
print("number of train images",nn)

dir_tcga = None
dir_tcga_project   = '/project/hikaku_db/data/tissue_images'
dir_tcga_local_mac = '/Users/nono/Documents/data/tissue_images'
if(os.path.exists(dir_tcga_project)):
    dir_tcga = dir_tcga_project
if(os.path.exists(dir_tcga_local_mac)):
    dir_tcga = dir_tcga_local_mac
if(dir_tcga == None):
    print('cannot find data dir')

list_dir_src = ("TCGA-05-4384-01A-01-BS1_files/15/",
                "TCGA-38-4631-11A-01-BS1_files/15/",
                "TCGA-05-4425-01A-01-BS1_files/15/")
ns = len(list_dir_src) # number of source images
file_imglist = 'filelist.txt'


dir_input_Users = '/Users/nono/Documents/data/tissue_images'
dir_input_home = '/home/nono/Documents/data/tissue_images'
if os.path.exists(dir_input_home) :
    dir_input = os.path.join(dir_input_home,'input_w{}'.format(nx))
if os.path.exists(dir_input_Users) :
    dir_input = os.path.join(dir_input_Users,'input_w{}'.format(nx))

if(not os.path.exists(dir_input)):
    os.makedirs(dir_input)

print(dir_input)

for aa in range(na):
    print(aa)
    qqq_src = []
    
    for ss,dir_src in enumerate(list_dir_src):
        qqq_ss = np.empty((nn, nx, ny, nl), np.float32)
        print(dir_src)
        fin = open(os.path.join(dir_tcga,dir_src,file_imglist),'r')
        table_train = list(csv.reader(fin, delimiter='\t'))
        fin.close()
        list_src = [x[0] for x in table_train]
        list_src_rand = np.random.choice(list_src, size=len(list_src), replace=False)
        
        ii = 0
        for file_src in list_src_rand:
            if(ii >= nn):
                break
            ## print(file_src)
            img_src = Image.open(os.path.join(dir_tcga,dir_src,file_src),'r')
        
            mx = img_src.size[0]
            my = img_src.size[1]
            for y0 in np.arange(0,my,sy):
                for x0 in np.arange(0,mx,sx):
                    if(ii >= nn):
                        break
                    if(x0+nx > mx or y0+ny > my):
                        continue
                    img_tmp = img_src.crop((x0,y0,x0+nx,y0+ny))
                    qqq_tmp = np.asarray(img_tmp) / 255.0

                    sd_tmp = np.std(np.asarray(img_tmp))
                    if sd_tmp < 16 :
                        continue # skip (almost) blank subframe
                    qqq_ss[ii,] = qqq_tmp
                    ii += 1
                # end for x0
            # end for y0
        # end for file_src
        print(ii)
        if(ii < nn):
            sys.exit('could not get enough slices {}/{}'.format(ii,nn))
        qqq_src.append(qqq_ss)
    # end of for ss

    qqq_src = np.vstack(qqq_src)
    np.save(os.path.join(dir_input,'qqq_trn_w{}_{}.npy'.format(nx,aa+1)), qqq_src)
# end of for aa
