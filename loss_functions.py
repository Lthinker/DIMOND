import torch
import qtlib
from matplotlib import pyplot as plt
import numpy as np
import sys
import time
import os
import pickle
# from torch_dmipy.utils import usedouble
import pdb
# import ricianloss
torch.cuda.set_device(int(os.environ["CUDA_USING"]))
# sys.path.append("/mnt/ACORN-base-MLP-b03D-manyFast/ACORN-base-MLP-b03DFast/pyDKI")
from pyDKI.dkimetrics import createTensorOrder

class FastgrdenlossMLP3D():
    def __init__(self,opt,Diso = 3.0e-3):
        # since all be div by 1000 -> Diso increase by 1000
        Diso = 3
        self.opt = opt
        self.cut = opt.useT1 + opt.useT2
        self.D_iso = torch.zeros([1,1,6])
        self.D_iso[...,0] = self.D_iso[...,3] = self.D_iso[...,5] = Diso
        self.D_iso[...,1] = self.D_iso[...,2] = self.D_iso[...,4] = 0
        self.D_iso = self.D_iso.to(torch.float32).cuda()
        self.elu = torch.nn.ELU(alpha=1)
        if opt.dkiconstrain:
            D_ind, D_cnt = createTensorOrder(2)
            W_ind, W_cnt = createTensorOrder(4)
            self.D_cnt = D_cnt
            self.D_ind = D_ind-1
            self.W_cnt = W_cnt
            self.W_ind = W_ind-1
            dir = np.loadtxt('/Data/Users/lzh/DIMOND/ACORN-base-MLP-NODDI/constraindir.txt')
            ndir = dir.shape[0]
            tensorconstrain = np.zeros([ndir,6])
            # kurtosisconstrain = self.W_cnt[np.ones([ndir,1]),:]*dir[:,self.W_ind[:,0]]*dir[:,self.W_ind[:,1]]*dir[:,self.W_ind[:,2]]*dir[:,self.W_ind[:,3]]
            kurtosisconstrain = np.repeat(self.W_cnt,ndir,axis=0)*dir[:,self.W_ind[:,0]]*dir[:,self.W_ind[:,1]]*dir[:,self.W_ind[:,2]]*dir[:,self.W_ind[:,3]]
            bconstrain = np.zeros([ndir,1])
            self.constrain = np.concatenate([tensorconstrain,kurtosisconstrain,bconstrain],axis=-1)
        pass
    def image_mse_diffusion(self,model_output, gt, step, tiling_every=100, dataset=None, model_type='multiscale', retile=True,trainchannel = None,validchannel = None):
        img_true = gt['img_block_coordpre']
        y_pred = model_output['model_out']['output']
        loss_weights = gt['entire_mask_block_coordpre']
        img_pred = -torch.matmul(y_pred.permute(0,2,1)[:,:,:6],gt['grden']).permute(0,2,1)
        img_pred = y_pred[:,6:,:]*torch.exp(img_pred)

        img_loss = ((img_true-img_pred)**2)*loss_weights # 231104
        return {'img_loss': img_loss.mean()}

   