import os
import nibabel as nb
import numpy as np
from matplotlib import pyplot as plt
from scipy import io as sio 
from numba import jit
from pyDKI.utils import *
import warnings
warnings.filterwarnings("ignore")
# @jit(cache=True)
def GenMatricsExample():
    dproot = '/mnt/ACORN-main/data/ManyPeople/parta/mwu105115'
    fpdt = os.path.join(dproot,'10subj-256h5conv3e1encode32Part2DKI','eval','pred_bestval.nii.gz')
    fpmask = os.path.join(dproot,'diff','mwu105115_diff_mask.nii.gz')
    dt = nb.load(fpdt).get_fdata()[50:60,50:60,69:70,:21]
    mask = np.expand_dims(nb.load(fpmask).get_fdata()[50:60,50:60,69:70],axis=-1)
    DTImaps, DKImaps = GenMatrics(dt,mask)
    return DTImaps,DKImaps
@jit(cache=True)
def GenMatrics(dt,mask):
    # dproot = '/mnt/ACORN-main/data/ManyPeople/parta/mwu105115'
    # fpdt = os.path.join(dproot,'10subj-256h5conv3e1encode32Part2DKI','eval','pred_bestval.nii.gz')
    # fpmask = os.path.join(dproot,'diff','mwu105115_diff_mask.nii.gz')
    # dt = nb.load(fpdt).get_fdata()[60:61,45:46,69:70,:21]
    # mask = np.expand_dims(nb.load(fpmask).get_fdata()[60:61,45:46,69:70],axis=-1)
    # print('dt.shape',dt.shape)
    # print('mask.shape',mask.shape)
    # plt.imshow(dt[:,:,80,1],'gray')
    n = dt.shape[3]
    x,y,z = dt.shape[0],dt.shape[1],dt.shape[2]
    assert len(dt.shape)==4, 'size need to be x y z 21+1'
    assert n==21, 'size need to be x y z 21+1'
    # dt = dt[np.repeat(mask,21,axis=-1)>0]\n"
    dt = dt[mask[:,:,:,0]>0,:].T
    nvoxels = dt.shape[1]
    DTIdict = DTImetrics(dt)
    dirs = sio.loadmat('/mnt/ACORN-base-MLP-b03D-manyFast/ACORN-base-MLP-b03DFast/dirs256.mat')['dirs']
    DKIdict = DKImetrics(parain={'dt':dt,'dirs':dirs},DTIdict=DTIdict)
    DTImaps = {}
    DKImaps = {}
    for m in ['l1','fa','md','fa']:
        tmp = np.zeros([x,y,z,1])
        tmp[mask>0] = DTIdict[m].T[:,0]
        DTImaps[m] = tmp
    for m in ['mk','ak','rk']:
        # print('m',m)
        # print(DKIdict[m])
        tmp = np.zeros([x,y,z,1])
        tmp[mask>0] = DKIdict[m].T[:,0]
        DKImaps[m] = tmp
    for m in ['e1']:
        tmp = np.zeros([x,y,z,3])
        tmp[np.repeat(mask>0,3,axis=-1)] = DTIdict[m].T.flatten()
        DTImaps[m] = tmp
    print('GenMatrix')
    return DTImaps,DKImaps
@jit(cache=True)
def DTImetrics(dt):
    # DTI \n",
    nvoxels = dt.shape[1]
    l1 = np.zeros([1, nvoxels])
    l2 = np.zeros([1, nvoxels])
    l3 = np.zeros([1, nvoxels])
    fa = np.zeros([1, nvoxels])
    e1 = np.zeros([3, nvoxels])
    e2 = np.zeros([3, nvoxels])
    e3 = np.zeros([3, nvoxels])
    for i in range(0,nvoxels):
        DT = dt[[0,1,2,1,3,4,2,4,5],i]
        DT = np.reshape(DT,[3,3])
        # try:
        [eigval,eigvec] = np.linalg.eig(DT)
        idx = np.argsort(-eigval)
        eigvec = eigvec[:,idx]
        eigval = eigval[idx]
        # except Exception:
            # eigvec = np.full([3,3],np.nan)
            # eigval = np.full([3,1],np.nan)
        # md = np.mean(eigval)
        # if np.sum(eigval*eigval)==0:
            # fa[0,i] = 0
        # else:
            # fa = np.sqrt(np.sum((eigval-md)*(eigval-md))) / np.sqrt(np.sum(eigval*eigval))*np.sqrt(1.5)
        l1[0,i] = eigval.flatten()[0]
        l2[0,i] = eigval.flatten()[1]
        l3[0,i] = eigval.flatten()[2]
        e1[:,i] = eigvec[:,0]
        e2[:,i] = eigvec[:,1]
        # e3[i,:] = eigvec[2]
    md = (l1+l2+l3)/3
    rd = (l2+l3)/2
    ad = l1
    fa = np.sqrt(1/2)*np.sqrt((l1-l2)**2+(l2-l3)**2+(l3-l1)**2)/np.sqrt(l1**2+l2**2+l3**2)
    DTImetrics = {}
    DTImetrics = {'l1':l1,'l2':l2,'l3':l3,
            'e1':e1, 'e2':e2, 'e3':e3,
            'md':md,'rd':rd,'ad':ad,'fa':fa}
    # return None
    return DTImetrics
@jit(cache=True)
def DKImetrics(parain,DTIdict):
    dt = parain['dt']
    dirs = parain['dirs']
    nvoxels = dt.shape[1]
    e1 = DTIdict['e1']
    akc,adc = AKC(dt,dirs)
    mk = np.array([np.mean(akc,axis=0)])
    ak = np.zeros([1,dt.shape[-1]])
    rk = np.zeros([1,dt.shape[-1]])
    for i in range(0,nvoxels):
        dirs = np.array(np.concatenate([e1[:,i:i+1],-e1[:,i:i+1]],axis=1)).T
        # print('dirs',dirs)
        akc,adc = AKC(dt[:,i:i+1],dirs)
        ak[:,i] = np.mean(akc)
        dirs = radialsampling(e1[:,i], 256).T
        akc,adc = AKC(dt[:,i:i+1], dirs)
        rk[:,i] = np.mean(akc)
    DKImetrics = {'mk':mk,'ak':ak,'rk':rk}
    return DKImetrics