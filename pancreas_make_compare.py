from PIL import Image

## dir_img = '/Users/nono/Documents/data/KPC/'

list_img_file_he = ('KPC F838-2/HE/KPC-F838-2_2015_08_26_0002.tif',6732, 1332)

list_img_file_ki = ('KPC F838-2/Ki67/KPC-F838-2_2015_10_05_0010.tif',6748, 1148)

xs,ys = 1024,1024

ff_src = list_img_file_he[0]
x0,y0 = list_img_file_he[1],list_img_file_he[2]
path_src = os.path.join(dir_image,ff_src)
img_src = Image.open(path_src,'r')
img_src_1K = img_src.crop((x0,y0,x0+xs,y0+ys))
# img_src_1K.save('pancreas_compare/img_src_1K_1.tif')

ff_dst = list_img_file_ki[0]
x1,y1 = list_img_file_ki[1],list_img_file_ki[2]
path_dst = os.path.join(dir_image,ff_dst)
img_dst = Image.open(path_dst,'r')
img_dst_1K = img_dst.crop((x1,y1,x1+xs,y1+ys))
# img_dst_1K.save('pancreas_compare/img_dst_1K_1.tif')

qqq_src_1K = np.asarray(img_src_1K) / 256.0
qqq_dst_1K = np.asarray(img_dst_1K) / 256.0

mx,my = img_src_1K.size    
nx = ny = 32
nb = 10000
qqq_src = np.zeros((nb,ny,nx,3))
qqq_dst = np.zeros((nb,ny,nx,3))
for bb in range(nb):
    x0 = np.random.choice(range(mx-nx),size=1)[0]
    y0 = np.random.choice(range(my-ny),size=1)[0]
    qqq_src[bb,] = qqq_src_1K[x0:(x0+nx),y0:(y0+ny),]
    qqq_dst[bb,] = qqq_dst_1K[x0:(x0+nx),y0:(y0+ny),]
        

np.save(os.path.join(dir_data,'pancreas_he_w32.npy'),qqq_src)
np.save(os.path.join(dir_data,'pancreas_ki_w32.npy'),qqq_dst)
