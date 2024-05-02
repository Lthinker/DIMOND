# qtlib.py
#
#
# (c) Qiyuan Tian, Harvard, 2021

import numpy as np
import nibabel as nb
from matplotlib import pyplot as plt
import torch
def block_ind(mask, sz_block=64, sz_pad=0):

    # find indices of smallest block that covers whole brain
    tmp = np.nonzero(mask);
    xind = tmp[0]
    yind = tmp[1]
    zind = tmp[2]
    
    xmin = np.min(xind); xmax = np.max(xind);
    ymin = np.min(yind); ymax = np.max(yind);
    zmin = np.min(zind); zmax = np.max(zind);
    ind_brain = [xmin, xmax, ymin, ymax, zmin, zmax]; 
    
    # calculate number of blocks along each dimension
    xlen = xmax - xmin + 1;
    ylen = ymax - ymin + 1;
    zlen = zmax - zmin + 1;
    
    nx = int(np.ceil(xlen / sz_block)) + sz_pad;
    ny = int(np.ceil(ylen / sz_block)) + sz_pad;
    nz = int(np.ceil(zlen / sz_block)) + sz_pad;
    
    # determine starting and ending indices of each block
    xstart = xmin;
    ystart = ymin;
    zstart = zmin;
    
    xend = xmax - sz_block + 1;
    yend = ymax - sz_block + 1;
    zend = zmax - sz_block + 1;
    
    xind_block = np.round(np.linspace(xstart, xend, nx));
    yind_block = np.round(np.linspace(ystart, yend, ny));
    zind_block = np.round(np.linspace(zstart, zend, nz));
    
    ind_block = np.zeros([xind_block.shape[0]*yind_block.shape[0]*zind_block.shape[0], 6])
    count = 0
    for ii in np.arange(0, xind_block.shape[0]):
        for jj in np.arange(0, yind_block.shape[0]):
            for kk in np.arange(0, zind_block.shape[0]):
                ind_block[count, :] = np.array([xind_block[ii], xind_block[ii]+sz_block-1, yind_block[jj], yind_block[jj]+sz_block-1, zind_block[kk], zind_block[kk]+sz_block-1])
                count = count + 1
    
    ind_block = ind_block.astype(int);
    
    return ind_block, ind_brain

def block_ind_min(mask, sz_block=64, sz_pad=0):

    # find indices of smallest block that covers whole brain
    tmp = np.nonzero(mask);
    zind = tmp[2]
    zmin = np.min(zind); zmax = np.max(zind);
    # calculate number of blocks along each dimension
    zlen = zmax - zmin + 1;
    
    nz = int(np.ceil(zlen / sz_block)) + sz_pad;
    
    zstart = zmin;
    zend = zmax - sz_block + 1;
    zind_block = np.round(np.linspace(zstart, zend, nz)).astype(int);
    
    ind_block = []
    for ii in range(0,len(zind_block)):
        ind_block_xy, ind_brain_xy = block_ind2D(mask=mask[:,:,zind_block[ii]:zind_block[ii]+sz_block-1], sz_block=sz_block,sz_pad=sz_pad)
        for jj in range(0,ind_block_xy.shape[0]):
            ind_block.append([ind_block_xy[jj][0],ind_block_xy[jj][1],ind_block_xy[jj][2],ind_block_xy[jj][3],zind_block[ii],zind_block[ii]+sz_block-1])
    # determine starting and ending indices of each block
    ind_block = np.array(ind_block).astype(int);
    
    return ind_block, None

def denormalize_image(imgall, imgnormall, mask):
    imgresall_denorm = np.zeros(imgall.shape)
    
    for jj in np.arange(imgall.shape[-1]):
        img = imgall[:, :, :, jj : jj + 1]
        imgres = imgnormall[:, :, :, jj : jj + 1]
        
        img_mean = np.mean(img[mask > 0.5])
        img_std = np.std(img[mask > 0.5])
    
        imgres_norm = (imgres * img_std + img_mean) * mask;
        
        imgresall_denorm[:, :, :, jj : jj + 1] = imgres_norm
    return imgresall_denorm
    
def normalize_image(imgall, imgresall, mask):
    imgall_norm = np.zeros(imgall.shape)
    imgresall_norm = np.zeros(imgall.shape)
    
    for jj in np.arange(imgall.shape[-1]):
        img = imgall[:, :, :, jj : jj + 1]
        imgres = imgresall[:, :, :, jj : jj + 1]
        
        img_mean = np.mean(img[mask > 0.5])
        img_std = np.std(img[mask > 0.5])
    
        img_norm = (img - img_mean) / img_std * mask;
        imgres_norm = (imgres - img_mean) / img_std * mask;
        
        imgall_norm[:, :, :, jj : jj + 1] = img_norm
        imgresall_norm[:, :, :, jj : jj + 1] = imgres_norm
    return imgall_norm, imgresall_norm
        
        
def extract_block(data, inds):
    xsz_block = inds[0, 1] - inds[0, 0] + 1
    ysz_block = inds[0, 3] - inds[0, 2] + 1
    zsz_block = inds[0, 5] - inds[0, 4] + 1
    ch_block = data.shape[-1]
    
    blocks = np.zeros((inds.shape[0], xsz_block, ysz_block, zsz_block, ch_block))
    
    for ii in np.arange(inds.shape[0]):
        inds_this = inds[ii, :]
        blocks[ii, :, :, :, :] = data[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :]
    
    return blocks
def extract_block_torch(data, inds):
    xsz_block = inds[0, 1] - inds[0, 0] + 1
    ysz_block = inds[0, 3] - inds[0, 2] + 1
    zsz_block = inds[0, 5] - inds[0, 4] + 1
    ch_block = data.shape[-1]
    
    # blocks = torch.zeros((inds.shape[0], xsz_block, ysz_block, zsz_block, ch_block),dtype=torch.float32)
    blocks = []
    for ii in np.arange(inds.shape[0]):
        inds_this = inds[ii, :]
        # blocks[ii, :, :, :, :] = torch.tensor(data[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :]).to(torch.float32)
        blocks.append(torch.tensor( data[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] ).permute(3,0,1,2).to(torch.float32))
    return blocks

def bvec2grden(bvec):
    grden = []
    for ii in range(0,bvec.shape[1]):
        gix = bvec[0,ii]
        giy = bvec[1,ii]
        giz = bvec[2,ii]
        grden.append([gix*gix,2*gix*giy,2*gix*giz,giy*giy,2*giy*giz,giz*giz])
    return np.array(grden)
def bvec2grdenKurtosis(bvec):
    grden = []
    for ii in range(0,bvec.shape[1]):
        gix = bvec[0,ii]
        giy = bvec[1,ii]
        giz = bvec[2,ii]
        grden.append([gix*gix,2*gix*giy,2*gix*giz,giy*giy,2*giy*giz,giz*giz])
    return np.array(grden)

def diffnormalization(bval,diff):
    b0 = np.zeros(diff[:,:,:,0:1].shape)
    b0count = 0
    for ii in range(0,len(bval)):
        # print(bval[ii])
        if abs(bval[ii])<50:
            b0 = b0 + diff[:,:,:,ii:ii+1]
            b0count = b0count+1
    b0 = b0/b0count
    b0[b0==0]=1 # avoid divide by 0
    masks = []
    diffb0norm = []
    diffnorm = []
    # for ii in range(0,len(bval)):
    #     diffb0norm.append(diff[:,:,:,ii:ii+1]/b0) # log e
    #     diffb0norm

    for ii in range(0,len(bval)):
        # diffnorm.append(np.log(diff[:,:,:,ii:ii+1]/b0) / (-bval[ii])) # log e that is we only compare the adc
        diffnorm.append(np.log(diff[:,:,:,ii:ii+1]/b0) )
        if bval[ii]==0:
            masks.append(np.zeros(diff[:,:,:,ii:ii+1].shape))
        else:
            masks.append(1-np.isnan(diffnorm[-1]))
    
    masks = np.concatenate(masks,-1)
    diffnorm = np.concatenate(diffnorm,-1)
    # return diffnorm,masks
    diffnorm[abs(diffnorm)>1e6]=0
    print("np.max(diffnorm)",np.max(diffnorm))
    return np.nan_to_num(diffnorm),masks
def diffnormalization_exp(bval,diff):
    b0 = np.zeros(diff[:,:,:,0:1].shape)
    b0count = 0
    for ii in range(0,len(bval)):
        # print(bval[ii])
        if abs(bval[ii])<50:
            b0 = b0 + diff[:,:,:,ii:ii+1]
            b0count = b0count+1
    b0 = b0/b0count
    b0[b0==0]=1
    masks = []
    diffnorm = []
    for ii in range(0,len(bval)):
        diffnorm.append(diff[:,:,:,ii:ii+1]/b0) # log e
        if abs(bval[ii])<=50:
            masks.append(np.zeros(diff[:,:,:,ii:ii+1].shape))
        else:
            masks.append( (diffnorm[-1]>0)*(diffnorm[-1]<1) )
    masks = np.concatenate(masks,-1)
    diffnorm = np.concatenate(diffnorm,-1)
    diffnorm[abs(diffnorm)>1]=0 # since it is decay
    print("np.max(diffnorm)",np.max(diffnorm))
    return diffnorm,masks
def mean_squared_error_weighted(y_true, y_pred):        
    loss_weights = y_true[0][:, :, :, :, -1:]
    imgs_true = y_true[0][:, :, :, :, :-1]
    grden = y_true[1]
    imgs_pred = tf.matmul(y_pred,grden)
    
    y_true_weighted = imgs_true * loss_weights
    y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights

    return K.mean(K.square(y_pred_weighted - y_true_weighted), axis=-1)
class grdenloss():
    def __init__(self,grden,img):
        self.grden = tf.convert_to_tensor(np.array(grden.T,dtype=np.float32) )
        print("self.grden",tf.shape(self.grden))
        self.gap = img.shape[-1]
        print("self.gap",self.gap)
    def grden_mean_squared_error_weighted(self,y_true, y_pred):        
        img_true = y_true[:,:,:,:,:self.gap]
        loss_weights = y_true[:,:,:,:,self.gap:]
        img_pred = tf.matmul(y_pred,self.grden)
        # print(tf.shape(img_pred))
        # print(tf.shape(img_true))

        y_true_weighted = img_true * loss_weights
        y_pred_weighted = img_pred * loss_weights

        return K.mean(K.square(y_pred_weighted - y_true_weighted), axis=-1)
class grdenloss_exp():
    def __init__(self,grden,img):
        print("not robustly process all bval")
        self.grden = tf.convert_to_tensor(np.array(grden.T,dtype=np.float32) )
        print("self.grden",tf.shape(self.grden))
        self.gap = img.shape[-1]
        print("self.gap",self.gap)
    def grden_mean_squared_error_weighted(self,y_true, y_pred):        
        img_true = y_true[:,:,:,:,:self.gap]
        loss_weights = y_true[:,:,:,:,self.gap:-1]
        b0scale_block = y_true[:,:,:,:,-1:]

        img_pred = tf.matmul(y_pred,self.grden)
        img_pred = -tf.sigmoid(img_pred)
        img_pred = b0scale_block*tf.exp(img_pred) 
        # print(tf.shape(img_pred))
        # print(tf.shape(img_true))

        y_true_weighted = img_true * loss_weights
        y_pred_weighted = img_pred * loss_weights

        return K.mean(K.square(y_pred_weighted - y_true_weighted), axis=-1)
    def grden_mean_squared_error_weighted_FT(self,y_true, y_pred):        
        img_true = y_true[:,:,:,:,:self.gap]
        loss_weights = y_true[:,:,:,:,self.gap:-1]
        b0scale_block = y_true[:,:,:,:,-1:]

        img_pred = tf.matmul(y_pred,self.grden)
        # img_pred = -tf.sigmoid(img_pred)
        img_pred = -img_pred
        img_pred = b0scale_block*tf.exp(img_pred) 
        # print(tf.shape(img_pred))
        # print(tf.shape(img_true))

        y_true_weighted = img_true * loss_weights
        y_pred_weighted = img_pred * loss_weights

        return K.mean(K.square(y_pred_weighted - y_true_weighted), axis=-1)
def mean_absolute_error_weighted(y_true, y_pred):
    loss_weights = y_true[:, :, :, :, -1:]
    y_true_weighted = y_true[:, :, :, :, :-1] * loss_weights
    y_pred_weighted = y_pred[:, :, :, :, :-1] * loss_weights
    
    return K.mean(K.abs(y_pred_weighted - y_true_weighted), axis=-1)

def block2brain(blocks, inds, mask):
    vol_brain = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], blocks.shape[-1]])
    vol_count = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], blocks.shape[-1]])
    
    for tt in np.arange(inds.shape[0]):
        inds_this = inds[tt, :]
        
        vol_brain[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] = \
                vol_brain[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] + blocks[tt, :, :, :, :]
        
        vol_count[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] = \
                vol_count[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] + 1.
    
    vol_count[vol_count < 0.5] = 1.
    vol_brain = vol_brain / vol_count 
    
    vol_brain = vol_brain * mask
    vol_count = vol_count * mask
    
    return vol_brain, vol_count 

def save_nii(fpNii, data, fpRef):
    
    new_header = header=nb.load(fpRef).header.copy()    
    new_img = nb.nifti1.Nifti1Image(data, None, header=new_header)    
    nb.save(new_img, fpNii)  



def block_ind2D(mask, sz_block=64, sz_pad=0):

    # find indices of smallest block that covers whole brain
    tmp = np.nonzero(mask);
    xind = tmp[0]
    yind = tmp[1]
    
    xmin = np.min(xind); xmax = np.max(xind)
    ymin = np.min(yind); ymax = np.max(yind)
    ind_brain = [xmin, xmax, ymin, ymax] 
    
    # calculate number of blocks along each dimension
    xlen = xmax - xmin + 1
    ylen = ymax - ymin + 1
    
    nx = int(np.ceil(xlen / sz_block)) + sz_pad
    ny = int(np.ceil(ylen / sz_block)) + sz_pad
    
    # determine starting and ending indices of each block
    xstart = xmin
    ystart = ymin

    xend = xmax - sz_block + 1
    yend = ymax - sz_block + 1
    
    xind_block = np.round(np.linspace(xstart, xend, nx))
    yind_block = np.round(np.linspace(ystart, yend, ny))
    
    ind_block = np.zeros([xind_block.shape[0]*yind_block.shape[0], 4])
    count = 0
    for ii in np.arange(0, xind_block.shape[0]):
        for jj in np.arange(0, yind_block.shape[0]):
                ind_block[count, :] = np.array([xind_block[ii], xind_block[ii]+sz_block-1, yind_block[jj], yind_block[jj]+sz_block-1])
                count = count + 1
    
    ind_block = ind_block.astype(int)
    
    return ind_block, ind_brain
def block2brain2D(blocks, inds, mask):
    vol_brain = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], blocks.shape[-1]])
    vol_count = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], blocks.shape[-1]])
    
    for tt in np.arange(inds.shape[0]):
        inds_this = inds[tt, :]
        
        vol_brain[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] = \
                vol_brain[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] + blocks[tt, :, :, :, :]
        
        vol_count[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] = \
                vol_count[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, inds_this[4]:inds_this[5]+1, :] + 1.
    
    vol_count[vol_count < 0.5] = 1.
    vol_brain = vol_brain / vol_count 
    
    vol_brain = vol_brain * mask
    vol_count = vol_count * mask
    
    return vol_brain, vol_count
def block2brain2DMLP(blocks, inds, mask):
    vol_brain = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], blocks.shape[-1]])
    vol_count = np.zeros([mask.shape[0], mask.shape[1], mask.shape[2], blocks.shape[-1]])
    
    for tt in np.arange(inds.shape[0]):
        inds_this = inds[tt, :]
        
        vol_brain[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, :] = \
                vol_brain[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1,  :] + np.expand_dims(blocks[tt, :, :, :],2)
        
        vol_count[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1,  :] = \
                vol_count[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1,  :] + 1.
    
    vol_count[vol_count < 0.5] = 1.
    vol_brain = vol_brain / vol_count 
    
    vol_brain = vol_brain * np.expand_dims(mask,3)
    vol_count = vol_count * np.expand_dims(mask,3)
    
    return vol_brain, vol_count
def extract_block2D(data, inds):
    xsz_block = inds[0, 1] - inds[0, 0] + 1
    ysz_block = inds[0, 3] - inds[0, 2] + 1
    ch_block = data.shape[-1]
    
    blocks = np.zeros((inds.shape[0], xsz_block, ysz_block, ch_block))
    
    for ii in np.arange(inds.shape[0]):
        inds_this = inds[ii, :]
        blocks[ii, :, :, :] = data[inds_this[0]:inds_this[1]+1, inds_this[2]:inds_this[3]+1, :]
    
    return blocks