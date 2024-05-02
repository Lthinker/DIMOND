# -*- coding: utf-8 -*-

"""
@Time    : 2022/9/26 14:30
@Author  : Zihan Li
@Email   : zihan-li18@outlook.com
"""

import numpy as np
import nibabel as nb
import pandas as pd
from matplotlib import pyplot as plt
from numba import jit
import qtlib
import warnings
import os
import glob
warnings.filterwarnings("ignore")

@jit(cache=True)
def Tensor2Para(model,mask=None):
    print("model.shape",model.shape)
    tensor_mtx = np.zeros([3,3],dtype = np.float64)
    dtimatrix = {}
    matrixes1D = ["L1","L2","L3","FA","MD","RD"]
    matrixes3D = ["V1","V2","V3"]
    for matrix in matrixes1D:
        dtimatrix[matrix] = np.zeros(model.shape[0:3])
    for matrix in matrixes3D:
        dtimatrix[matrix] = np.zeros(list(model.shape[0:3])+[3])
    D = np.zeros([3])
    V = np.zeros([3,3])
    # for ii in range(60,model.shape[0]):
    #     for jj in range(40,model.shape[1]):
    #         for kk in range(30,model.shape[2]):
    for ii in range(0,model.shape[0]):
        for jj in range(0,model.shape[1]):
            for kk in range(0,model.shape[2]):
                if mask is not None and mask[ii,jj,kk,0]==0:
                    continue
                mpoint = model[ii][jj][kk]
                tensor_mtx[0,0] = mpoint[0]
                tensor_mtx[0,1] = mpoint[1];tensor_mtx[1,0] = mpoint[1];
                tensor_mtx[0,2] = mpoint[2];tensor_mtx[2,0] = mpoint[2];
                tensor_mtx[1,1] = mpoint[3]
                tensor_mtx[1,2] = mpoint[4];tensor_mtx[2,1] = mpoint[4];
                tensor_mtx[2,2] = mpoint[5]
                [tmpD,tmpV] = np.linalg.eig(tensor_mtx) # the V has been sum
                s2l = np.argsort(tmpD) # from small number to large number
                for ss in range(0,3):
                    D[ss] = tmpD[s2l[ss]]
                    V[:,ss] = tmpV[:,s2l[ss]]
                # print("tensor_mtx",tensor_mtx)
                # print("D",D)
                # print("V",V)
                # print("V1",V[:,2])
                # print(D) # [1.10513204 0.76443632 0.41076772]
                # return
                MD = np.mean(D)
                if np.sum(D*D)==0:
                    FA = 1000
                else:
                    FA = np.sqrt(np.sum( (D-MD)*(D-MD) )) / np.sqrt(np.sum(D*D))*np.sqrt(1.5)
                RD = np.mean(D[1:])
                
                dtimatrix["V1"][ii,jj,kk] = V[:,2]
                dtimatrix["V2"][ii,jj,kk] = V[:,1]
                dtimatrix["V3"][ii,jj,kk] = V[:,0]

                dtimatrix["L1"][ii,jj,kk] = D[2]
                dtimatrix["L2"][ii,jj,kk] = D[1]
                dtimatrix["L3"][ii,jj,kk] = D[0]


                dtimatrix["FA"][ii,jj,kk] = FA
                dtimatrix["MD"][ii,jj,kk] = MD
                dtimatrix["RD"][ii,jj,kk] = RD
                # return 0,0
                # print("V.shape:",V.shape,";D.shape",D.shape)
    return dtimatrix,model
    
def calc_ang(vol1,vol2):
    norm1 = np.sqrt(vol1[:,:,:,0:1]*vol1[:,:,:,0:1]+vol1[:,:,:,1:2]*vol1[:,:,:,1:2]+vol1[:,:,:,2:3]*vol1[:,:,:,2:3])
    vol1 = vol1/norm1
    norm2 = np.sqrt(vol2[:,:,:,0:1]*vol2[:,:,:,0:1]+vol2[:,:,:,1:2]*vol2[:,:,:,1:2]+vol2[:,:,:,2:3]*vol2[:,:,:,2:3])
    vol2 = vol2/norm2
    dotprod = abs(np.sum(vol1*vol2,axis=-1))
    dotprod[dotprod>1] = 1
    return np.arccos(dotprod)/np.pi*180

def Tensor2Para2File(modelpath,root,prefix,fpref,mask = None):
    model = nb.load(modelpath).get_fdata()
    assert(len(model.shape)==4)
    if mask is not None:
        assert len(mask.shape)==4
    if model.shape[-1]==7:
        model = model[:,:,:,:6]
    # print("model before",model[60,60,70])
    model = model/1000
    # print("model after",model[60,60,70])
    dtimatrix,_ = Tensor2Para(model,mask = mask)
    dpfile = os.path.join(root,prefix)
    if os.path.exists(dpfile):
        pass
    else:
        os.mkdir(dpfile)
    # print("dpfile",dpfile)
    for metric in ["V1","V2","V3","MD","RD","FA","L1"]:
        qtlib.save_nii(os.path.join(dpfile,os.path.basename(prefix)+"_"+metric+".nii.gz"),dtimatrix[metric],fpref)
    # print("finish!")


def EvalDti(dproot,CalPrefix,RefPrefix,tmask,metrics=["L1","FA","MD","V1"]):
    error = {}
    for ii in range(0,len(metrics)):
        ref = nb.load(os.path.join(dproot,RefPrefix,os.path.basename(RefPrefix) +"_"+ metrics[ii] +".nii.gz")).get_fdata()
        cal = nb.load(os.path.join(dproot,CalPrefix,os.path.basename(CalPrefix) +"_"+ metrics[ii] +".nii.gz")).get_fdata()
     
        # print("ref.shape",ref.shape)
        # print("cal.shape",cal.shape)

        if metrics[ii] in ["V1","V2","V3"]:
            # print("ref.shape",ref.shape)            
            tmp = calc_ang(ref,cal)
            nanmask = np.isnan(tmp)
            infmask = np.isinf(tmp)
            btmask = tmask>0.5
            rmmask  = (nanmask+infmask)>0.5
            angerr = np.nanmean(tmp[:,:,:][btmask[:,:,:] & (~rmmask[:,:,:])])
            error[metrics[ii]] = angerr
            # print(metrics[ii]+" mean angular error: ",angerr)
        else:
            nanmask = ()
            mae = np.nanmean(abs(ref-cal)[tmask>0])
            if metrics[ii] in ["MD","RD","L1"]:
                mae=mae*1000
            error[metrics[ii]] = mae
            # print(metrics[ii] + " mae: ",mae)
    return error

def EvalDti_block(dproot,CalPrefix,RefPrefix,wholemask,metrics=["L1","FA","MD","V1"],ind_block=[60,80,60,80,0,1]):
    error = {}
    # print("ind_block",ind_block)
    for ii in range(0,len(metrics)):
        ref = nb.load(os.path.join(dproot,RefPrefix,os.path.basename(RefPrefix) +"_"+ metrics[ii] +".nii.gz")).get_fdata()
        cal = nb.load(os.path.join(dproot,CalPrefix,os.path.basename(CalPrefix) +"_"+ metrics[ii] +".nii.gz")).get_fdata()

        ref = ref[ind_block[0]:ind_block[1],ind_block[2]:ind_block[3],ind_block[4]:ind_block[5]]
        cal = cal[ind_block[0]:ind_block[1],ind_block[2]:ind_block[3],ind_block[4]:ind_block[5]]
        # print(wholemask.shape)
        tmask = wholemask[ind_block[0]:ind_block[1],ind_block[2]:ind_block[3],ind_block[4]:ind_block[5]]
        # print("ref.shape",ref.shape)
        # print("cal.shape",cal.shape)
        # print("tmask.shape",tmask.shape)
        if metrics[ii] in ["V1","V2","V3"]:
            tmp = calc_ang(ref,cal)
            nanmask = np.isnan(tmp)
            infmask = np.isinf(tmp)
            btmask = tmask>0.5
            rmmask  = (nanmask+infmask)>0.5
            angerr = np.nanmean(tmp[:,:,:][btmask[:,:,:] & (~rmmask[:,:,:])])
            error[metrics[ii]] = angerr
            # print(metrics[ii]+" mean angular error: ",angerr)
        else:
            nanmask = ()
            mae = np.nanmean(abs(ref-cal)[tmask>0])
            if metrics[ii] in ["MD","RD","L1"]:
                mae=mae*1000
            error[metrics[ii]] = mae
            # print(metrics[ii] + " mae: ",mae)
    return error

def EvalDtiNew(dproot,fpCal,fpRef,tmask,metrics=["L1","FA","MD","V1"]):
    error = {}
    for ii in range(0,len(metrics)):
        ref = nb.load(os.path.join(fpRef,os.path.basename(fpRef) +"_"+ metrics[ii] +".nii.gz")).get_fdata()
        cal = nb.load(os.path.join(dproot,fpCal,os.path.basename(fpCal) +"_"+ metrics[ii] +".nii.gz")).get_fdata()
        # print("ref.shape",ref.shape)
        # print("cal.shape",cal.shape)
        if metrics[ii] in ["V1","V2","V3"]:
            # print("ref.shape",ref.shape)            
            tmp = calc_ang(ref,cal)
            nanmask = np.isnan(tmp)
            infmask = np.isinf(tmp)
            btmask = tmask>0.5
            rmmask  = (nanmask+infmask)>0.5
            angerr = np.nanmean(tmp[:,:,:][btmask[:,:,:] & (~rmmask[:,:,:])])
            error[metrics[ii]] = angerr
            # print(metrics[ii]+" mean angular error: ",angerr)
        else:
            nanmask = ()
            mae = np.nanmean(abs(ref-cal)[tmask>0])
            if metrics[ii] in ["MD","RD","L1"]:
                mae=mae*1000
            error[metrics[ii]] = mae
            # print(metrics[ii] + " mae: ",mae)
    return error

def EvalDti_blockNew(dproot,fpCal,fpRef,wholemask,metrics=["L1","FA","MD","V1"],ind_block=[0,64,0,64,0,64]):
    error = {}
    # print("ind_block",ind_block)
    for ii in range(0,len(metrics)):
        ref = nb.load(os.path.join(dproot,fpRef,os.path.basename(fpRef) +"_"+ metrics[ii] +".nii.gz")).get_fdata()
        cal = nb.load(os.path.join(dproot,fpCal,os.path.basename(fpCal) +"_"+ metrics[ii] +".nii.gz")).get_fdata()
        ref = ref[ind_block[0]:ind_block[1],ind_block[2]:ind_block[3],ind_block[4]:ind_block[5]]
        cal = cal[ind_block[0]:ind_block[1],ind_block[2]:ind_block[3],ind_block[4]:ind_block[5]]
        # print(wholemask.shape)
        tmask = wholemask[ind_block[0]:ind_block[1],ind_block[2]:ind_block[3],ind_block[4]:ind_block[5]]
        # print("ref.shape",ref.shape)
        # print("cal.shape",cal.shape)
        # print("tmask.shape",tmask.shape)
        if metrics[ii] in ["V1","V2","V3"]:
            tmp = calc_ang(ref,cal)
            nanmask = np.isnan(tmp)
            infmask = np.isinf(tmp)
            btmask = tmask>0.5
            rmmask  = (nanmask+infmask)>0.5
            angerr = np.nanmean(tmp[:,:,:][btmask[:,:,:] & (~rmmask[:,:,:])])
            error[metrics[ii]] = angerr
            # print(metrics[ii]+" mean angular error: ",angerr)
        else:
            nanmask = ()
            mae = np.nanmean(abs(ref-cal)[tmask>0])
            if metrics[ii] in ["MD","RD","L1"]:
                mae=mae*1000
            error[metrics[ii]] = mae
            # print(metrics[ii] + " mae: ",mae)
    return error
def b0rescalenum(dproot,diff='diff'):
    bval = pd.read_csv(glob.glob(os.path.join(dproot,diff,"*.bval"))[0],header=None).values.T[0]
    bvec = pd.read_csv(glob.glob(os.path.join(dproot,diff,"*.bvec"))[0],header=None,sep=" ").values.T
    bvecsqsum = np.sum(bvec*bvec,axis=0)
    bvecsqsum[bvecsqsum==0]=1
    bvec = bvec/np.array([bvecsqsum for ss in range(0,3)])
    fpImg = glob.glob(os.path.join(dproot,diff,"*_diff.nii.gz"))[0]
    fpMask = glob.glob(os.path.join(dproot,'aparc',"MASK_BRAIN_TISSUE.nii.gz"))[0]# not b0rescale using tissue mask
         
    diff = nb.load(fpImg).get_fdata()   
    mask = nb.load(fpMask).get_fdata()
    b0 = np.zeros(diff[:,:,:,0:1].shape)
    b0count = 0
    for ii in range(0,len(bval)):
        # print(bval[ii])
        if abs(bval[ii])<50:
            b0 = b0 + diff[:,:,:,ii:ii+1]
            b0count = b0count+1
    b0 = b0/b0count
    scalenumber = np.percentile(b0[mask>0], 50)
    return scalenumber
def tensor2DWI(dpSub, model,b0):
    bval = pd.read_csv(glob.glob(os.path.join(dpSub,"*.bval"))[0],header=None).values.T
    bvec = pd.read_csv(glob.glob(os.path.join(dpSub,"*.bvec"))[0],header=None,sep=" ").values.T
    bvecsqsum = np.sum(bvec*bvec,axis=0)
    bvecsqsum[bvecsqsum==0]=1
    bvec = bvec/np.array([bvecsqsum for ss in range(0,3)])
    grden = qtlib.bvec2grden(bvec)
    grden = grden.T
    # print("model.shape",model.shape)
    # print("b0.shape",b0.shape)
    img_pred = b0*np.exp(-np.matmul(model,grden))
    return img_pred
def MCDropIntefgrate(fpmask,dproot,label,numrep=20):
    preds = []
    mask = np.expand_dims(nb.load(fpmask).get_fdata(),-1)
    for ii in range(0,numrep):
        fppred = os.path.join(dproot,f"pred{ii:03d}_{label}.nii.gz")
        preds.append(np.expand_dims(nb.load(fppred).get_fdata()*mask,0))
    predavg = np.mean(preds,0)[0,:,:,:,:]
    qtlib.save_nii(os.path.join(dproot,f'pred_{label}Dropout.nii.gz'),predavg,"/mnt/ACORN-main/data/MRIfile/mwu100307_diff.nii.gz")
    