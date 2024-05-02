import math
import os
import errno
import matplotlib.colors as colors
import torch
from PIL import Image
from torch.utils.data import Dataset
# from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import urllib.request
from tqdm import tqdm
import numpy as np
import copy
# from inside_mesh import inside_mesh
import nibabel as nb
from scipy.spatial import cKDTree as spKDTree
import qtlib
import pandas as pd
import glob
import time
from matplotlib import pyplot as plt
import glob
from pyDKI.utils import *
import time
from numba import jit
import utils
from scipy.ndimage import binary_dilation
torch.cuda.set_device(int(os.environ["CUDA_USING"]))


def to_uint8(x):
    return (255. * x).astype(np.uint8)


def to_numpy(x):
    return x.detach().cpu().numpy()

@jit(cache=True)
def diff_divscale_exp_noddi(bval,diff,mask,opt):
    # print('bval',bval)
    b0 = np.zeros(diff[:,:,:,0:1].shape)
    b0count = 0
    entire_mask = []
    for ii in range(0,len(bval)):
        # print(bval[ii])
        if abs(bval[ii])<20:
            b0 = b0 + diff[:,:,:,ii:ii+1]
            b0count = b0count+1
    b0 = b0/b0count

    b0_div = b0.copy()
    b0_div[b0_div==0]=1
    for ii in range(0,len(bval)):
        tmpmask = np.ones(diff[:,:,:,ii:ii+1].shape)
        if opt.b0process == 2 and bval[ii]<25:
            tmpmask = np.zeros(diff[:,:,:,ii:ii+1].shape)
        entire_mask.append(mask*tmpmask)
        print(np.sum(entire_mask[-1]))
    scalenumber = np.percentile(b0[mask>0], 50)
    diffnorm = diff/scalenumber
    b0scale = b0/scalenumber
    print("scalenumber",scalenumber)
    entire_mask = np.concatenate(entire_mask,-1)
    return diffnorm,b0scale,entire_mask,scalenumber

@jit(cache=True)
def diff_divscale_exp_noddi_new(bval,diff,mask,opt):
    # print('bval',bval)
    b0 = np.zeros(diff[:,:,:,0:1].shape)
    b0count_map = np.zeros(diff[:,:,:,0:1].shape)
    entire_mask = []
    signal_min_threshold = 2e-16
    for ii in range(0,len(bval)):
        # print(bval[ii])
        if abs(bval[ii])<25:
            diff_tmp = diff[:,:,:,ii:ii+1] 
            b0 = b0 + diff_tmp
            b0count_map = b0count_map + (diff_tmp>signal_min_threshold)
    b0count_map_copy = b0count_map.copy()
    b0count_map[b0count_map==0]=1
    b0 = b0/b0count_map
    

    b0_div = b0.copy()
    b0_div[b0_div==0]=1
    for ii in range(0,len(bval)):
        diff_tmp = diff[:,:,:,ii:ii+1] 
        if opt.b0process == 2 and bval[ii]<25:
            tmpmask = np.zeros(diff_tmp.shape)
        else:
            tmpmask = np.ones(diff_tmp.shape)
        tmpmask[diff_tmp < signal_min_threshold] = 0
        entire_mask.append(mask*tmpmask*(b0count_map_copy>0) ) # b0全是0的点要remove掉
        print(np.sum(entire_mask[-1]))
    # print('b0.shape',b0.shape,'mask.shape',mask.shape)
    scalenumber = np.percentile(b0[mask>0], 50)
    diffnorm = diff/scalenumber
    b0scale = b0/scalenumber
    print("scalenumber",scalenumber)
    entire_mask = np.concatenate(entire_mask,-1)
    return diffnorm,b0scale,entire_mask,scalenumber,b0count_map_copy

@jit(cache=True)
def conv3D(img,kernel):
    convresult = np.zeros(img.shape)
    for ii in range(1,img.shape[0]-1):
        for jj in range(1,img.shape[1]-1):
            for kk in range(1,img.shape[2]-1):
                convresult[ii,jj,kk,0] = np.sum(img[ii-1:ii+2,jj-1:jj+2,kk-1:kk+2,0]*kernel)
    return convresult
def localsum_image(img):
    kernel = np.zeros([3,3,3])
    kernel[0,1,1] = 1
    kernel[1,0,1] = 1
    kernel[1,1,0] = 1
    kernel[1,1,2] = 1
    kernel[1,2,1] = 1
    kernel[2,1,1] = 1
    kernel = kernel
    avgmask = np.zeros(img.shape)
    for jj in range(0,img.shape[-1]):
        avgmask[:,:,:,jj:jj+1] = conv3D(img[:,:,:,jj:jj+1],kernel)
    return avgmask,kernel
def avgmaskfill_1b0(img_norm,b0scale,entire_mask ,b0scalenum,bval):
    img_norm_sum,_ = localsum_image(img_norm*entire_mask)
    entire_mask_sum,_ = localsum_image(entire_mask)
    entire_mask_sum[entire_mask_sum==0]=1
    img_norm_blur = img_norm_sum/entire_mask_sum
    img_norm_avg = img_norm*entire_mask+img_norm_blur*(1-entire_mask)
    # make the b0 b0
    img_norm_avg[:,:,:,0] = img_norm_avg[:,:,:,0] + img_norm[:,:,:,0]
    return img_norm_avg,b0scale,entire_mask ,b0scalenum
def avgmaskfill(img_norm,b0scale,entire_mask ,b0scalenum,bval):
    img_norm_sum,_ = localsum_image(img_norm*entire_mask)
    entire_mask_sum,_ = localsum_image(entire_mask)
    entire_mask_sum[entire_mask_sum==0]=1
    img_norm_blur = img_norm_sum/entire_mask_sum
    img_norm_avg = img_norm*entire_mask+img_norm_blur*(1-entire_mask)
    # make the b0 b0
    for ii in range(0,img_norm.shape[-1]):
        if abs(bval[ii]-0)<20:
            img_norm_avg[:,:,:,ii] = img_norm_avg[:,:,:,ii] + img_norm[:,:,:,ii]
    return img_norm_avg,b0scale,entire_mask ,b0scalenum
def diff_mulrescale_exp(bval,diff,diffnorm,mask):
    b0 = np.zeros(diff[:,:,:,0:1].shape) 
    b0count = 0
    for ii in range(0,len(bval)):
        # print(bval[ii])
        if abs(bval[ii])<50:
            b0 = b0 + diff[:,:,:,ii:ii+1]
            b0count = b0count+1
    b0 = b0/b0count
    scalenumber = np.percentile(b0[mask>0], 50)
    diffnorm_back = diffnorm*scalenumber
    return diffnorm_back

class NiftiFile3D_block_NODDI(Dataset):
    def __init__(self, dpSub, grayscale=True,use_tmask = False,useT1=False,useT2=False,aug=0,opt=None):
        super().__init__()
        start = time.time()
        bval = pd.read_csv(glob.glob(os.path.join(dpSub,opt.subpath,"*.bval"))[0],header=None).values.T
        bvec = pd.read_csv(glob.glob(os.path.join(dpSub,opt.subpath,"*.bvec"))[0],header=None,sep=" ").values.T   


        # process bvec
        bvecsqsum = np.sum(bvec*bvec,axis=0)
        bvecsqsum[bvecsqsum==0]=1
        bvec = bvec/np.array([bvecsqsum for ss in range(0,3)])

        fpImg = glob.glob(os.path.join(dpSub,opt.subpath,"*_diff.nii.gz"))[0]
        if use_tmask:
            fpMask = glob.glob(os.path.join(dpSub,'aparc',"MASK_BRAIN_TISSUE.nii.gz"))[0]
        else:
            print("brainmask")
            fpMask = glob.glob(os.path.join(dpSub,opt.subpath, "*_mask.nii.gz"))[0]
            
        img = nb.load(fpImg).get_fdata()    
        mask = nb.load(fpMask).get_fdata()
        self.imageshape = img.shape[0:3]
        if opt.sz_block_mode in ['min32']:
            mask = binary_dilation(mask)     
        mask = mask * 1.0 
        mask = np.expand_dims(mask, 3)
        print("loading data time",time.time()-start)
        img_norm,b0scale,entire_mask,b0scalenum,b0count_map_copy = diff_divscale_exp_noddi_new(bval[0],img,mask,opt)
    
        img_norm[img_norm<=0] = 2e-16
        self.img = img_norm
        self.mask = mask
        self.entire_mask = entire_mask

        self.b0scale = b0scale
        self.b0scalenum = b0scalenum
        print("normalize data time",time.time()-start)

        
        # generate grad
        if opt.dki_weighted:
            b = GenDkib(bval,bvec,opt)
            b = np.concatenate([b,np.ones([1,b.shape[1]])],axis = 0)
            pinvb = np.linalg.pinv(b)
            faltimg_norm = img_norm.reshape([-1,img_norm.shape[-1]])
            olsmodel = np.log(faltimg_norm) @ pinvb
            olssignal = np.exp(olsmodel @ b)
            reflatimg_norm = olssignal.reshape(img_norm.shape)    
            # reflatimg_norm = np.exp(reflatimg_norm)
            print("b.shape",b.shape)
            self.grden = torch.tensor(b).to(torch.float32) 
            self.pinvgrden = torch.tensor(pinvb).to(torch.float32)

        else:
            if opt.dataset in ['DWI3DManyFast10','budadata ']:
                b = qtlib.bvec2grden(bvec).T
                self.grden = torch.tensor(b*bval/1000).to(torch.float32)
            else:
                b = GenDkib(bval,bvec,opt)
                self.grden = torch.tensor(b).to(torch.float32)
            
        print("weighted data time",time.time()-start)

        # separate block
        if opt.sz_block_mode == 'min32':
            ind_block, ind_brain = qtlib.block_ind_min(self.mask, sz_block=opt.sz_block, sz_pad=1)
            # double check the ind_block is good
            tmp_mask_block = qtlib.extract_block(self.mask,ind_block)
            tmp_mask_block_sum = np.array([np.sum(item) for item in tmp_mask_block])
            tmp_mask_block_index = np.where(tmp_mask_block_sum!=0)[0]
            ind_block = ind_block[tmp_mask_block_index]
        else:
            ind_block, ind_brain = qtlib.block_ind(self.mask, sz_block=opt.sz_block, sz_pad=0)

        self.ind_block = ind_block
        self.img_block = qtlib.extract_block(self.img,ind_block)
        self.mask_block = qtlib.extract_block(self.mask,ind_block)
        self.entire_mask_block = qtlib.extract_block(self.entire_mask,ind_block)

            
        self.b0scale_block = qtlib.extract_block(self.b0scale,ind_block)
        print("divide block time",time.time()-start)

        if aug == 0:
            pass
        elif aug == 2:
            self.mask_block = np.concatenate([self.mask_block,np.flip(self.mask_block,axis=1)],axis=0)
            self.img_block = np.concatenate([self.img_block,np.flip(self.img_block,axis=1)],axis=0)
            self.entire_mask_block = np.concatenate([self.entire_mask_block,np.flip(self.entire_mask_block,axis=1)],axis=0)
            self.b0scale_block = np.concatenate([self.b0scale_block,np.flip(self.b0scale_block,axis=1)],axis=0)
        else:
            assert(0)
       
        print("data augmentation time",time.time()-start)
        

        def mask_block2coord(mask_block):
            mask_block_coord = []
            for ii in range(0,mask_block.shape[0]):
                zerocoords = np.where(mask_block[ii]==1)
                mask_block_coord.append(zerocoords[0:3])
            return mask_block_coord
        def img_block2coord(img_block,mask_block_coords):
            
            img_block_coordpre = []
            for ii in range(0,len(mask_block_coords) ):
                coord0 = mask_block_coords[ii][0]
                coord1 = mask_block_coords[ii][1]
                coord2 = mask_block_coords[ii][2]
                img_block_coordpre.append(torch.tensor(img_block[ii,coord0,coord1,coord2]).to(torch.float32).permute(1,0) )
            return img_block_coordpre
        start = time.time()
        self.mask_block_coords = mask_block2coord(self.mask_block)
        self.NoneEdgeIndex = utils.NoneEdge(self.mask_block_coords,opt)

              

        self.b0scale_block_coordpre = img_block2coord(self.b0scale_block,self.mask_block_coords)
        print('b0',time.time()-start)

        tmpblock = self.img_block
        self.img_block_coordpre = img_block2coord(tmpblock[:,:,:,:,:tmpblock.shape[-1]-useT1-useT2],self.mask_block_coords)
        print('img block coord',time.time()-start)
        self.entire_mask_block_coordpre = img_block2coord(self.entire_mask_block,self.mask_block_coords)        

        self.mask_block_coordpre = img_block2coord(self.mask_block,self.mask_block_coords)

        self.img_block = torch.tensor(self.img_block).to(torch.float32)
        self.mask_block = torch.tensor(self.mask_block).to(torch.float32)
        self.entire_mask_block = torch.tensor(self.entire_mask_block).to(torch.float32)
        self.b0scale_block = torch.tensor(self.b0scale_block).to(torch.float32)
        
        self.img_block = self.img_block.permute(0,4,1,2,3)
        self.mask_block = self.mask_block.permute(0,4,1,2,3)
        self.entire_mask_block = self.entire_mask_block.permute(0,4,1,2,3)
        self.b0scale_block = self.b0scale_block.permute(0,4,1,2,3)

        self.kd_tree_sp = None # not sure
        self.img_channels = 1
        self.length = self.img_block.shape[0]
        
        self.supplementary = {'bvec':torch.tensor(bvec).float()}
        if not (opt is None) and opt.convnetwork in [14,15,16]:
            self.supplementary = {'bvec':torch.tensor( np.concatenate([bvec,bval/1000],axis=0)        ).float()}
        # load default parameters

        

        if len(bval.shape)==2 and bval.shape[0]==1:
            bval = bval[0]
        S0 = np.expand_dims(np.mean(self.img[:,:,:,bval<20],-1),-1)
        S0_block = qtlib.extract_block(S0, ind_block)
        if aug == 0:
            pass
        elif aug == 2:
            S0_block = np.concatenate([S0_block,np.flip(S0_block,axis=1)],axis=0)
        else:
            assert(0)
        S0_block_coord = img_block2coord(S0_block,self.mask_block_coords)
        self.supplementary['S0_block_coord'] = S0_block_coord

        self.supplementary['NoneEdgeIndex'] = self.NoneEdgeIndex



    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.img

class FastImageNormalDataset3D(torch.utils.data.Dataset):
    def __init__(self, dataset, patch_size=(16, 16), sidelength=None, random_coords=False,
                 jitter=True, num_workers=0, length=1000, scale_init=3, max_patches=1024,opt=None):

        # handle parallelization
        self.num_workers = num_workers
        self.dataset = dataset        
        self.length = self.dataset.length
        self.img = self.dataset[0]
        self.ind_block = self.dataset.ind_block
        self.mask = self.dataset.mask
        self.b0scale = self.dataset.b0scale
        self.b0scalenum = self.dataset.b0scalenum
        self.jitter = jitter
        self.eval = False
        self.opt = opt
        if opt.MCchannel:
            self.bvalnotzero = self.dataset.bvalnotzero
        if opt.dataset in ['NODDI','DWI3DManyFast10','budadata']:
            self.scheme_hcp = None
            
        
        self.supplementary = dataset.supplementary
    def toggle_eval(self):
        if not self.eval:
            self.jitter_bak = self.jitter
            self.jitter = False
            self.eval = True
        else:
            self.jitter = self.jitter_bak
            self.eval = False

    def __len__(self):
        # return len(self.dataset)
        return self.length

    def __getitem__(self, idx):
        if not self.opt is None and self.opt.EarlyStopping:
            in_dict = {
                        'img_block': self.dataset.img_block[idx][:-1,:,:,:], # torch.Size([17, 64, 64, 64])
                        'mask_block': self.dataset.mask_block[idx],# torch.Size([1, 64, 64, 64])
                        'b0scale_block': self.dataset.b0scale_block[idx],
                        'mask_block_coords0': torch.tensor(self.dataset.mask_block_coords[idx][0]),# 130120,)
                        'mask_block_coords1': torch.tensor(self.dataset.mask_block_coords[idx][1]),
                        'mask_block_coords2': torch.tensor(self.dataset.mask_block_coords[idx][2])
                    }
        else:
            in_dict = { 
                        'img_block': self.dataset.img_block[idx],
                        'mask_block': self.dataset.mask_block[idx],
                        'mask_block_coords0': torch.tensor(self.dataset.mask_block_coords[idx][0]),
                        'mask_block_coords1': torch.tensor(self.dataset.mask_block_coords[idx][1]),
                        'mask_block_coords2': torch.tensor(self.dataset.mask_block_coords[idx][2]),
                        'bvec': self.supplementary['bvec']
                    }

        gt_dict = {'img_block_coordpre': self.dataset.img_block_coordpre[idx],
                   'b0scale_block_coordpre':self.dataset.b0scale_block_coordpre[idx],
                   'mask_block_coordpre':self.dataset.mask_block_coordpre[idx],
                   'entire_mask_block_coordpre':self.dataset.entire_mask_block_coordpre[idx],
                   'grden': self.dataset.grden}
        try:
            gt_dict['F_block_coordpre'] = self.dataset.F_block_coordpre[idx]
        except Exception as err:
            pass


        if not self.opt is None:
            if self.opt.loaddefault in [4]:
                gt_dict['S0_block_coord'] = self.dataset.supplementary['S0_block_coord'][idx]

            if self.opt.sz_block_mode in ['min32']:
                gt_dict['NoneEdgeIndex'] = self.dataset.NoneEdgeIndex[idx]
        return in_dict, gt_dict

class PartANODDIFT(Dataset):
    def __init__(self, dpSubs, subj,grayscale=True,use_tmask = False,fploadlist = None,useT1 = False,useT2 = False,aug=0,opt=None):
        super().__init__()
        self.img_datasets = []
        self.idxdict = {}
        numdataset = 0
        count = 0
        self.grdenlist = []

        dpSub = os.path.join(dpSubs,subj)
        img_dataset = FastImageNormalDataset3D(NiftiFile3D_block_NODDI(dpSub,grayscale=grayscale,use_tmask = use_tmask, aug=aug,opt=opt),opt=opt)
        self.b0scalenum = img_dataset.b0scalenum
        self.img_datasets.append(img_dataset)
        for ii in range(0,img_dataset.length):
            self.idxdict[count+ii]=[numdataset,ii,dpSub]
        numdataset = numdataset + 1
        count = count + img_dataset.length
        print(f"******{numdataset}*******")

        self.length = count
        self.scheme_hcp = img_dataset.scheme_hcp
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        searchidx = self.idxdict[idx]
        return self.img_datasets[searchidx[0]][searchidx[1]]
