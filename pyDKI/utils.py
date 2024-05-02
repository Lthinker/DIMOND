import numpy as np
import math
from numba import jit
@jit(cache=True)
def nchoosek(matrix,order):
    lens = matrix.shape[-1]
    indexes = []
    initlist = [i for i in range(0,order)]
    indexes.append(initlist.copy())
    orderstate = order
    # print('lens',lens)
    while(1):
        if len(initlist)>0:
            popval = initlist.pop()
        else:
            break
        if popval>=lens-1:
            if len(indexes) == 0:
                break
            else:
                continue
        pp=1
        if (popval+order-len(initlist))<lens:
            while(len(initlist)<order):
                initlist.append(popval+pp)
                pp = pp+1
            indexes.append(initlist.copy())          
        # print(indexes)
    # print('matrix.shape:',matrix.shape)
    # print('indexes[0]',indexes[0])
    # print(matrix)
    # print(matrix[0,indexes])
    return matrix[0,indexes]
@jit(cache=True)
def createTensorOrder(order):
    m1 = np.array([1,2,3])
    m2 = np.ones([1,order])
    kronresult = np.kron(m1,m2)
    X = nchoosek(kronresult,order)
    X = np.unique(X,axis=0)
    # print(X.shape)
    cnt = []
    for i in range(0,X.shape[0]):
        # print(math.factorial(order))
        # print(math.factorial(np.sum(X[i]==1)))
        cnt.append(math.factorial(order)/math.factorial(np.sum(X[i]==1))/math.factorial(np.sum(X[i]==2))/math.factorial(np.sum(X[i]==3)))
    result = np.array([1])
    # result = np.array(X,dtype=np.int),np.array([cnt])
    result = np.array(X).astype(int),np.array([cnt])
    return result
@jit(cache=True)
def GenDkib(bval,bvec,opt):
    # print('bval',bval)
    # print('bvec',bvec)
    # 20230506注释
    if opt.dki_weighted:
        bval_dwi = bval.T # use b0 but mask them in loss
        bvec_dwi = bvec.T # use b0 but mask them in loss
    else:
        bval_dwi = np.array([bval[bval>500]]).T
        bvec_dwi = bvec.T[(bval>500)[0]]
    

    # print('bval_dwi',bval_dwi)
    # print('bvec_dwi',bvec_dwi)
    grad = np.concatenate([bvec_dwi,bval_dwi/1000],axis=1)
    bval = grad[:,3]
    ndwis = len(bval)
    D_ind, D_cnt = createTensorOrder(2)
    W_ind, W_cnt = createTensorOrder(4)
    # print('grad',grad)
    # print('bval',bval)
    # print('D_ind',D_ind) 
    bD = np.repeat(D_cnt,ndwis,axis=0)*grad[:,D_ind[:,0]-1]*grad[:,D_ind[:,1]-1]
    bW = np.repeat(W_cnt,ndwis,axis=0)*grad[:,W_ind[:,0]-1]*grad[:,W_ind[:,1]-1]*grad[:,W_ind[:,2]-1]*grad[:,W_ind[:,3]-1]
    b = np.concatenate([-np.repeat(np.array([bval]).T,6,axis=1)*bD,np.repeat(np.array([bval]).T,15,axis=1)**2/6*bW],axis=1)
    # b = np.array([-np.repeat(bval,6,axis=1)*bD,])
    # print('np.repeat(D_cnt,ndwis,axis=0)',np.repeat(D_cnt,ndwis,axis=0))
    # print('D_ind[:,0]-1',D_ind[:,0]-1)
    # print('D_ind[:,1]-1',D_ind[:,1]-1)
    # print('grad[:,D_ind[:,0]-1]',grad[:,D_ind[:,0]-1])
    # print('grad[:,D_ind[:,1]-1]',grad[:,D_ind[:,1]-1])
    # print('bD',bD.shape)
    # print(bD)
    # print('bW',bW.shape)
    # print(bW)
    return b.T
@jit(cache=True)
def AKC(dt,dirs):
    W_ind, W_cnt = createTensorOrder(4)
    adc = ADC(dt[0:6, :], dirs)
    md = np.sum(dt[[0, 3, 5],:],0)/3
    # print('md',md)
    ndir  = dirs.shape[0]
    T =  np.repeat(W_cnt,ndir,axis=0)*dirs[:,W_ind[:,0]-1]*dirs[:,W_ind[:,1]-1]*dirs[:,W_ind[:,2]-1]*dirs[:,W_ind[:,3]-1]
    akc =  T@dt[6:21, :]
    # print('np.repeat(W_cnt,ndir,axis=0)',np.repeat(W_cnt,ndir,axis=0).shape,' ',np.repeat(W_cnt,ndir,axis=0))
    # print('dirs[:,W_ind[:,0]-1][0]',dirs[:,W_ind[:,0]-1][0])
    # print('W_ind[:,0]-1',W_ind[:,0]-1 )
    # print('T',T[0])
    # print('dirs[0]',dirs[0])
    # print('dirs[0][W_ind[:,0]-1]',dirs[0][W_ind[:,0]-1])
    # print('adc.shape',adc.shape,'akc.shape',akc.shape)
    # print('md.shape',md.shape,'md**2.shape',(md**2).shape)
    akc = (akc * np.tile(md**2, [adc.shape[0], 1]))/(adc**2)
    # print('akc',akc)
    # print('akc.shape',akc.shape)
    # print('md.shape',md.shape)
    return akc,adc
@jit(cache=True)
def ADC(dt,dirs):
    D_ind, D_cnt = createTensorOrder(2)
    ndir  = dirs.shape[0]
    T =  np.repeat(D_cnt,ndir,axis=0)*dirs[:,D_ind[:,0]-1]*dirs[:,D_ind[:,1]-1]
    adc = T @ dt
    return adc
@jit(cache=True)
def radialsampling(dir, n):
    dt = 2*np.pi/n
    theta = np.linspace(start=0,stop=2*np.pi-dt,num=n)
    dirs = [np.cos(theta).T, np.sin(theta), 0*theta.T]
    v = np.array([-dir[1], dir[0], 0])
    s = np.sqrt(np.sum(v**2))
    c = dir[2]
    V = np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]],[-v[1], v[0], 0]])
    # print('V',V)
    R = np.eye(3) + V + V@V * (1-c)/s**2
    # print('R',R)
    # print('c',c,'s',s)
    dirs = R@dirs
    return dirs

