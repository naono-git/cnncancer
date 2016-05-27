import os
import re
import pickle
import numpy as np
from PIL import Image
import datetime 

##def exists(varname):
##    print(len(globals()))
##    return(varname in locals())

def timestamp():
    time_cur = datetime.datetime.now()
    print('datetime:',time_cur.strftime('%m/%d %H:%M'))
    stamp = time_cur.strftime('%Y%m%d%H%M')
    return(stamp)

def saveObject(xxx,filename="neko.pkl",dirname='out1'):
    if(not os.path.exists(dirname)):
        os.makedirs(dirname)
    file_out = open(os.path.join(dirname,filename),'wb')
    pickle.dump(xxx,file_out)
    file_out.close()
    print(filename)

def getRemoteFile(filename,dirname='Documents/cnnspot/out1',hostname='127.0.0.1',portnum='20052',dst='out1'):
    if(portnum==None):
        portoption = ''
    else:
        portoption = '-P {}'.format(portnum)
    if(isinstance(filename,(list,tuple))):
        for ff in filename:
            getRemoteFile(ff,dirname,hostname,portnum,dst)
    if(isinstance(filename,str)):
        pathname = os.path.join(dirname,filename)
        if(not os.path.exists(pathname)):
            cmd = 'scp {} {}:{} {}'.format(portoption,hostname,pathname,dst)
            print(cmd)
            os.system(cmd)


def openRemoteImage(filename,dirname='Documents/cnnspot/out1',hostname='127.0.0.1',portnum='20052',dst='out1'):
    if(portnum==None):
        portoption = ''
    else:
        portoption = '-P {}'.format(portnum)
    pathname = os.path.join(dirname,filename)
    if(not os.path.exists(pathname)):
        cmd = 'scp {} {}:{} {}'.format(portoption,hostname,pathname,dst)
        print(cmd)
        os.system(cmd)
    cmd = 'open {}/{}'.format(dst,filename)
    os.system(cmd)

def cbind_image(img1, img2):
    new_size = (img1.size[0]+img2.size[0], np.max([img1.size[1],img2.size[1]]))
    new_im = Image.new('RGB', new_size)
    new_im.paste(img1, (0,0))
    new_im.paste(img2, (img1.size[0],0))
    return(new_im)

def rbind_image(img1, img2):
    new_size = (np.max([img1.size[0],img2.size[0]]),img1.size[1]+img2.size[1])
    new_im = Image.new('RGB', new_size)
    new_im.paste(img1, (0,0))
    new_im.paste(img2, (0,img1.size[1]))
    return(new_im)

def showsave(img,file_img,dir_img='out1'):
    path_img = os.path.join(dir_img,file_img)
    img.save(path_img, format='JPEG')
    print(path_img)
    if(re.search('xquartz',os.getenv('DISPLAY'))):
        img.show()
    return(file_img)
    
def get_index_from_ii(iii,shape):
    nn = len(iii)
    nd = len(shape)
    jjj = np.zeros((nn,nd),dtype='int')
    for bb,ii in enumerate(iii):
        for aa in range(nd):
            jjj[bb,nd-aa-1] = ii % shape[nd-aa-1]
            ii = ii // shape[nd-aa-1]
    return(jjj)
    
def which_array(bbb):
    shape_bbb = bbb.shape
    nn = np.sum(bbb)
    nd = len(shape_bbb)
    iii = [ii for ii, bb in enumerate(bbb.reshape(-1)) if bb]
    jjj = np.empty((nn,nd),dtype='int')
    for bb,ii in enumerate(iii):
        for aa in range(nd):
            jjj[bb,nd-aa-1] = ii % shape_bbb[nd-aa-1]
            ii = ii // shape_bbb[nd-aa-1]
    return(jjj)

def argmax_array(xxx):
    xshape = xxx.shape
    nd = len(xshape)
    ii = np.argmax(xxx)
    iii = np.empty(nd,dtype='int')
    for aa in range(nd):
        iii[nd-aa-1] = ii % xshape[nd-aa-1]
        ii = ii // xshape[nd-aa-1]
    return(tuple(iii))
    
