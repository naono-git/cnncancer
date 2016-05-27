import os
import re
import numpy as np
from PIL import Image

rwb = np.empty((256,3),dtype='uint8')
rwb[0:128,0]=255
rwb[0:128,1]=np.arange(0,255,2)
rwb[0:128,2]=np.arange(0,255,2)
rwb[128:256,0]=255-np.arange(0,255,2)
rwb[128:256,1]=255-np.arange(0,255,2)
rwb[128:256,2]=255

def get_RWB_from_qqq(qqq,qlim=None):
    if(qlim==None):
        qmin=np.min(qqq)
        qmax=np.max(qqq)
        qlim=(qmin,qmax)
        q0 = qlim[0]
        qr = qlim[1]-qlim[0]
    ny,nx = qqq.shape
    if(qr==0):
        rgb=np.zeros((ny,nx,3),dtype='uint8')
        rgb[:,:,0] = 255
        return(rgb)
    else:
        qqq = (qqq-q0) / qr * 255.0
        np.clip(qqq,0,255,qqq)
        rgb = rwb[np.asarray(qqq,dtype='uint8')]
        return(rgb)
    

def get_image_from_encode(qqq,qlim=None):
    if(len(qqq.shape)==4):
        nn,ny,nx,nl = qqq.shape
    if(len(qqq.shape)==3):
        nn = 1
        ny,nx,nl = qqq.shape
        qqq = qqq[np.newaxis,:,:,:]
    if(len(qqq.shape)==2):
        nn = 1
        nl = 1
        ny,nx = qqq.shape
        qqq = qqq[np.newaxis,:,:,np.newaxis]
    img_size = (nx*nn, ny*nl)
    img = Image.new('RGB',img_size)
    for aa in range(nn):
        for bb in range(nl):
            rgb = get_RWB_from_qqq(qqq[aa,:,:,bb])
            img.paste(Image.fromarray(rgb),(nx*aa,ny*bb))
    return img

# assume that qqq is a [0,1) rgb density
def get_image_from_qqq(qqq):
    if(len(qqq.shape)==4):
        nn,ny,nx,nl = qqq.shape
    if(len(qqq.shape)==3):
        nn = 1
        ny,nx,nl = qqq.shape
        qqq = qqq[np.newaxis,:,:,:]
    rgb = np.empty((nn,ny,nx,3),np.uint8)
    rgb[:] = np.clip(qqq[:]*255,0,255)
    rgb = rgb.transpose((1,0,2,3)) # nn,ny,nx,nl -> ny,nx,nn,nl
    rgb = rgb.reshape((ny,nx*nn,3))# row of nn images
    img = Image.fromarray(rgb)
    return img

# assume that pix is a [0,255] rgb pixel
def get_image_from_pix(pix):
    if(len(pix.shape)==3) :
        pix = pix[np.newaxis,]
    ni,nw,nh,nlayer = pix.shape
    aaa = np.zeros((nh*ni, nw, nlayer),np.uint8)
    h0 = 0
    for ii in range(0,ni):
        aaa[h0:h0+nh,:] = pix[ii,:]
        h0 += nh
    img = Image.fromarray(aaa)
    return img


def soft_entropy(xxx):
    ppp = np.absolute(xxx)
    ooo = np.sum(ppp,axis=3)+1e-10
    rrr = ppp / ooo[:,:,:,np.newaxis]
    sss = np.sum(-rrr * np.log(rrr),axis=3)
    return(sss)

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

