 # Enable import from parent package
import sys
import os
import configargparse
from scipy.ndimage import binary_dilation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
p = configargparse.ArgumentParser()
p.add('-c', '--config', required=False, is_config_file=True, help='Path to config file.')
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'   # using for debugging
# General training options
p.add_argument('--lr', type=float, default=1e-3, help='learning rate. default=1e-3')
p.add_argument('--num_iters', type=int, default=100300,
               help='Number of iterations to train for.')
p.add_argument('--num_workers', type=int, default=0,
               help='number of dataloader workers.')
p.add_argument('--skip_logging', action='store_true', default=True,
               help="don't use summary function, only save loss and models")
p.add_argument('--eval', action='store_true', default=False,
               help='run evaluation')
p.add_argument('--resume', nargs=3, type=str, default=None,
               help='resume training, specify path to directory where model is stored and the iteration of ckpt.')
p.add_argument('--resumekind', type= int, default=0,
               help='kind of resume to utilize')
p.add_argument('--gpu', type=int, default=2,
               help='GPU ID to use')
p.add_argument('--conv1', type=int, default=1,
               help='input conv size')
p.add_argument('--useT1T2loss', type=int, default=0,
               help='whether to use the T1T2loss')
p.add_argument('--numMC', type=int, default=20,
               help='whether to use the T1T2loss')
p.add_argument('--losstype', type=str, default="L2",
               help='whether to use the T1T2loss')
p.add_argument('--optimizer', type=str, default="Adam",
               help='which optimizer to use')
p.add_argument('--cv', type=int, default=0,
               help='whether using the cross validation')
p.add_argument('--EarlyStopping', type=int, default=0,
               help='whether using the early stopping')
p.add_argument('--MCchannel', type=int, default=0,
               help='whether using the early stopping')

# logging options
p.add_argument('--seed', type=int, required=True,default=-1,
               help='the random seed of the torch and numpy.')
p.add_argument('--experiment_name', type=str, required=True,
               help='the directory name of this experiment.')
p.add_argument('--epochs_til_ckpt', type=int, default=2,
               help='Epochs until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=500,
               help='Number of iterations until tensorboard summary is saved.')
p.add_argument('--overgpu', type=int, default=0,
               help='whether to calculate all the data on gpu')
p.add_argument('--savedutraining', type=int, default=0,
               help='whether to save during training')
# dataset options
p.add_argument('--res', nargs='+', type=int, default=[512],
               help='image resolution.')
p.add_argument('--dataset', type=str, default='camera', choices=['camera', 'pluto', 'tokyo', 'mars','DWI','1isodata','DWI3D','DWI3DManyFast','DWI3DManyFast10','DKI','Real3DWI','Fake3DWI','CDMD','CDMDWater','CDMDDTI',"CDMDDKI",'DWI3DManyFast10Arb1000','HighResoDWI','HighResoDWIsmall','DWI3DManyFast10Arb1000Sim',"NODDI","NODDI2DKI","budadata"],
               help='which dataset to use')
p.add_argument('--dataset_path', type=str, default='0',
               help='specific the path to the dataset')
p.add_argument('--grayscale', action='store_true', default=False,
               help='whether to use grayscale')
p.add_argument('--use_tmask', type = int, default = 0, 
               help='whether to use tissue mask for network training')
p.add_argument('--useT1', type = int, default = 0, 
               help='whether to use T1w for network training')
p.add_argument('--useT2', type = int, default = 0, 
               help='whether to use T2w for network training')
p.add_argument('--trainval', type = int, default = 0, 
               help='whether to split the training and validation set')
p.add_argument('--subj', type = str, 
               help='specify which subject is used for subj specific finetune')
p.add_argument('--direapply', type = int, default = 0, 
               help='whether apply the pretrain model, if 1, change the savedir')
p.add_argument('--numsubj', type = int, default = -1, 
               help='how many sub used')
p.add_argument('--numb1000', type = int, default = 15, 
               help='how many b1000 is used')
p.add_argument('--noiselevel', type = int, default = 0, 
               help='the noiselevel of simulated data')
p.add_argument('--subpath', type = str, default = "dki", 
               help='only works for dki')
p.add_argument('--aug', type = int, default = 2, 
               help='only works for dki')
p.add_argument('--dki_weighted', type = int, default = 0, 
               help='0: no weighted 1: weighted, hard weighted 2: weighted, softweighted ')
p.add_argument('--dki_constraintype', type = int, default = 0, 
               help='0: relu(-constrain)*relu(-constrain) 1: no constrain 2:relu(-constrain)')
p.add_argument('--dki_constrainstrength', type = int, default = 1, 
               help='the strength of the regularization')
p.add_argument('--dkiconstrain', type=int, default=0,
               help='whether to constrain the Kapp>0')
p.add_argument('--odiorder', type=int, default=5,
               help='whether to constrain the Kapp>0')
p.add_argument('--lossactfun', type=str, default='sigmoid',
               help='the activation function of loss function')

# model options
p.add_argument('--patch_size', nargs='+', type=int, default=[32],
               help='patch size.')
p.add_argument('--hidden_features', type=int, default=512,
               help='hidden features in network')
p.add_argument('--encoding_features', type=int, default=0,
               help='features for the feature encoder in network')
p.add_argument('--hidden_layers', type=int, default=4,
               help='hidden layers in network')
p.add_argument('--num_encoding_layers', type=int, default=1,
               help='encoding layers in network')
p.add_argument('--w0', type=int, default=5,
               help='w0 for the siren model.')
p.add_argument('--steps_til_tiling', type=int, default=500,
               help='How often to recompute the tiling, also defines number of steps per epoch.')
p.add_argument('--max_patches', type=int, default=1024,
               help='maximum number of patches in the optimization')
p.add_argument('--model_type', type=str, default='multiscale', required=False, choices=['multiscale', 'siren', 'pe','CNN','CNN3D','FastCNN3D'],
               help='Type of model to evaluate, default is multiscale.')
p.add_argument('--convnetwork', type=int, default=0, 
               help='whether to use the smooth convblock')
p.add_argument('--DropRate', type=float, default=0,  # before 20230828, default value is 0.1
               help='the dropout rate')
p.add_argument('--MCInference', type=int, default=0, 
               help='whether to use the MC inference')
p.add_argument('--lock', type=int, default=0, 
            help='only the last layer can be trained')
p.add_argument('--Vnoddi', type=str, default='dmipy', 
            help='the version of noddi model')
p.add_argument('--fusion', type=int, default=0, 
            help='kind of feature fusion')
p.add_argument('--diffusionmodel', type=str, default='tensor', 
            help='the forward model')

p.add_argument('--sz_block_mode', type=str, default='normal', 
            help='the number of output features of position embeding') # only set for NODDI
p.add_argument('--sz_block', type=int, default=64, 
            help='the number of output features of position embeding')
p.add_argument('--flateninput', type=int, default=0, 
            help='which type of input to be used')
p.add_argument('--aggregatemode', type=str, default='softmaxmean', 
            help='which type of input to be used')
p.add_argument('--Hyp1f1_dict', type=int, default=0, 
            help='which type of input to be used')
p.add_argument('--num_postconv', type=int, default=0, 
            help='the number of post convolution to be used')
p.add_argument('--init_with_other', type=int, default=0, 
            help='whether init the network with pre-defined result')
p.add_argument('--b0process', type=int, default=0, 
            help='0: no process; 1: bval[bval<25]=0; 2: not optimize')
p.add_argument('--representation', type=int, default=0, 
            help='0: not ultilize hash representation; 1: utilize hash representation')

# parameter for implicit representation
# the hash encoding
p.add_argument('--trainingstrate', type=int, default=0, 
            help='0: default, 1: with the learning rate decay...')
p.add_argument('--NODDIspecific', type=str, default='default', 
            help='default: using the HCPrawdiffusionData')
p.add_argument('--num_level', type=int, default=8, 
            help='the dimension of the feature of each level')
p.add_argument('--base_resolution', type=int, default=2, 
            help='the base resolution')
p.add_argument('--log2_hashmap_size', type=int, default=19, 
            help='size of hashmap')
p.add_argument('--desired_resolution', type=int, default=256, 
            help='desire resotion')
# the following network
p.add_argument('--hidden_dims', type=int, default=64, 
            help='the hidden features of the following network')


# saving
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--logging_root', type=str, default='../logs', help='root for logging')
p.add_argument('--DropoutType', type=str, default='MC', help='root for logging')
p.add_argument('--ngroups', type=int, default=1, help='root for logging')

# debug setting
p.add_argument('--loaddefault', type=int, default=0, help='whether to load default parameters')




opt = p.parse_args()

if opt.convnetwork in [1000]:
    assert(opt.aug==0)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
# os.environ["CUDA_LAUNCH_BLOCKING"] = str(opt.gpu)
os.environ["CUDA_USING"] = str(opt.gpu)
os.environ["ODI_ORDER"] = str(opt.odiorder)
os.environ["Hyp1f1_dict"] = str(opt.Hyp1f1_dict)

# if not os.path.join()
import dataio
import utils
import training
import pydti
import loss_functions
import pruning_functions
import modules
from torch.utils.data import DataLoader
from functools import partial
import numpy as np
# import skimage
# import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import torch
from time import time
import qtlib
import glob
import nibabel as nb
import random
import timm
from torch import nn

torch.cuda.set_device(int(os.environ["CUDA_USING"]))

if not opt.seed == -1:
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
# if opt.representation == 1:
opt.bound = 1
for k, v in opt.__dict__.items():
    print(k, v)


def main():
    assert(opt.MCchannel + opt.EarlyStopping<2)
    if len(opt.patch_size) == 1:
        opt.patch_size = 3*opt.patch_size

 
    if opt.eval:
        aug=0
        # aug = opt.aug
    else:
        aug=opt.aug
        # aug = 0
    # need to be specified
    dpdataset = f'./data/{opt.dataset_path}'    
    coord_dataset = dataio.PartANODDIFT(dpdataset,subj = opt.subj,useT1=opt.useT1,useT2=opt.useT2,aug=aug,use_tmask=opt.use_tmask,opt=opt)# add tmask
    fpref = glob.glob(os.path.join(os.path.join(dpdataset,opt.subj),opt.subpath,"*_diff.nii.gz"))[0]
    in_features = nb.load(fpref).shape[-1]
    opt.fpref = fpref
    utils.cond_mkdir(os.path.join(os.path.join(dpdataset,opt.subj,opt.experiment_name)))
    opt.num_epochs = opt.num_iters // coord_dataset.__len__()

    image_resolution = (opt.res, opt.res)
    if opt.eval:
        shuffle = False
    else:
        shuffle = True

    num_workers = opt.num_workers
    trainloader = timm.data.loader.MultiEpochsDataLoader(coord_dataset, shuffle=shuffle, batch_size=1, pin_memory=True,
                    num_workers=num_workers)
                
    if opt.resume is not None:
        path, iter, bestval = opt.resume
        iter = int(iter)
        print(path)
        assert(os.path.isdir(path))
        assert opt.config is not None, 'Specify config file'


    out_features = utils.noutfeas(opt)
    opt.in_features = in_features
    opt.out_features = out_features
    opt.imgshape = coord_dataset.img_datasets[0].img.shape[0:3]

    model = utils.get_model(opt,in_features,out_features)

    model.cuda(int(os.environ["CUDA_USING"]))
    
    # print number of model parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Num. Parameters: {params}')

    # Define the loss
    loss_fn = utils.DeterLoss(opt,coord_dataset)
    summary_fn = partial(utils.write_image_patch_multiscale_summary, image_resolution, opt.patch_size[1:], coord_dataset, model_type=opt.model_type, skip=opt.skip_logging)

    # Define the pruning function
    pruning_fn = partial(pruning_functions.no_pruning,
                         pruning_every=1)

    # if we are resuming from a saved checkpoint
    if opt.resume is not None:
        if bestval == 'bestval':
            print('Loading checkpoints')
            model_dict = torch.load(path + '/checkpoints/' + f'best_val_model.pth')
            model.load_state_dict(model_dict)
        elif bestval in ['1','10'] :
            pass
        elif bestval == 'ckbest':
            model_dict = torch.load(path + '/checkpoints/' + 'ck0_val_model.pth')
            model.load_state_dict(model_dict)

        elif iter!=0:    
            print('Loading checkpoints')
            model_dict = torch.load( glob.glob(path + '/checkpoints/' + f'*best_val_model_{iter:06d}.pth')[0] )
            model.load_state_dict(model_dict)
        elif iter ==0:
            print('training from scratch')
        else:
            assert(0)
        # load optimizers
        try:
            resume_checkpoint = {}
            optim_dict = torch.load(path + '/checkpoints/' + f'optim_{iter:06d}.pth')
            for g in optim_dict['optimizer_state_dict']['param_groups']:
                g['lr'] = opt.lr
            resume_checkpoint['optimizer_state_dict'] = optim_dict['optimizer_state_dict']
            resume_checkpoint['total_steps'] = optim_dict['total_steps']
            resume_checkpoint['epoch'] = optim_dict['epoch']

        except FileNotFoundError:
            print('Unable to load optimizer checkpoints')
    else:
        resume_checkpoint = {}


    if opt.eval:
        run_evalFast_dropout(model, coord_dataset,opt.resume[2],numiter = f"{iter:06d}",direapply=opt.direapply,numMC = opt.numMC)
    else:
        root_path = os.path.join(os.path.join(dpdataset,opt.subj,opt.experiment_name))
        utils.cond_mkdir(root_path)
        if opt.resume is None:
            pass
        else:
            if bestval!='bestval' and bestval!='ckbest':
                root_path = os.path.join(os.path.join(dpdataset,opt.subj,opt.experiment_name,f'FT{iter:06d}'))
                utils.cond_mkdir(root_path)
            
        p.write_config_file(opt, [os.path.join(root_path, 'config.ini')])

        # Save text summary of model into log directory.
        with open(os.path.join(root_path, "model.txt"), "w") as out_file:
            out_file.write(str(model))

        training.trainMCvalidationLight(model=model, train_dataloader=trainloader, epochs=opt.num_epochs, lr=opt.lr,
                    steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,optimizer = opt.optimizer,
                    model_dir=root_path, loss_fn=loss_fn, pruning_fn=pruning_fn, summary_fn=summary_fn, objs_to_save=None,
                    resume_checkpoint=resume_checkpoint,opt=opt,coord_dataset = coord_dataset)



def run_evalFast_dropout(model, coord_dataset,bestval,numiter = -1,direapply=0,numMC=20):
    fpref = opt.fpref
    # get checkpoint directory
    checkpoint_dir = os.path.join(os.path.dirname(opt.config), 'checkpoints')

    # make eval directory
    eval_dir = os.path.join(os.path.dirname(opt.config), 'eval')
    utils.cond_mkdir(eval_dir)

    # get model & optim files
    if bestval == '0':
        model_files = [os.path.basename(glob.glob(os.path.join(checkpoint_dir,f"*model_{int(numiter):06d}.pth") )[0])]
        optim_files = []

    elif bestval == '10':
        model_files = [os.path.basename(glob.glob(os.path.join(checkpoint_dir,f"epoch{int(numiter):06d}.pth") )[0])]
        optim_files = []

    else:
        model_files = sorted([f for f in os.listdir(checkpoint_dir) if re.search(r'1best_val_model_[0-9]+.pth', f)], reverse=True)
        optim_files = sorted([f for f in os.listdir(checkpoint_dir) if re.search(r'1best_val_optim_[0-9]+.pth', f)], reverse=True)

    # extract iterations
    print('model_files,',model_files)
    if bestval in ['10']:
        iters = [int(re.findall(r'[0-9]+', f)[0]) for f in model_files]
    else:
        iters = [int(re.findall(r'[0-9]+', f)[1]) for f in model_files]
    print("*************",iters,"********************")
    # append beginning of path
    model_files = [os.path.join(checkpoint_dir, f) for f in model_files]
    optim_files = [os.path.join(checkpoint_dir, f) for f in optim_files]

    # iterate through model and optim files
    metrics = {}
    saved_gt = False
    start = time()
    # print(f"infinfinf{inf:03d}")
    for curr_iter, model_path in zip(tqdm(iters), model_files):
        pred_curriter = []
        if bestval=='bestval':
            print('Loading bestval models')
            model_path = os.path.join(checkpoint_dir,'best_val_model.pth')
            model_dict = torch.load(model_path)
            model.load_state_dict(model_dict)
        elif bestval == 'ckbest':
            print('Loading bestval models')
            model_path = os.path.join(checkpoint_dir,'ck0_val_model.pth')
            model_dict = torch.load(model_path)
            model.load_state_dict(model_dict)
        else:
            print('Loading models')
            print("model_path",model_path)
            model_dict = torch.load(model_path)
            # initialize model state_dict
            print('Initializing models')
            model.load_state_dict(model_dict)
        for inf in range(0,numMC):
            preds = []
            idxdict = coord_dataset.idxdict

            dpSub = idxdict[0][2]
            print('dpSub',dpSub)
            print('bestval here',bestval)
            b0rescale = coord_dataset.b0scalenum

            for ii in range(0,coord_dataset.length):
                model_input, gt = coord_dataset[ii]
                idxdict = coord_dataset.idxdict

                # convert to cuda and add batch dimension
                tmp = {}
                for key, value in model_input.items():
                    if isinstance(value, torch.Tensor):
                        tmp.update({key: value[None, ...].cuda(int(os.environ["CUDA_USING"]))})
                    else:
                        tmp.update({key: value})
                model_input = tmp

                tmp = {}
                for key, value in gt.items():
                    if isinstance(value, torch.Tensor):
                        tmp.update({key: value[None, ...].cuda(int(os.environ["CUDA_USING"]))})
                    else:
                        tmp.update({key: value})
                gt = tmp
                print('Running forward pass')

                model.eval()
                def apply_dropout(m): 
                    if type(m) == nn.Dropout: m.train()
                model.apply(apply_dropout)
                
                with torch.no_grad():
                    pred_imgline = model(model_input)['model_out']['output']

                torch.cuda.synchronize()
                print(f'Model: {time() - start:.02f}')
                pred_2img = torch.zeros([model_input['img_block'].shape[0],model_input['img_block'].shape[2],model_input['img_block'].shape[3],model_input['img_block'].shape[4],pred_imgline.shape[1] ]) 
                coord0 = model_input['mask_block_coords0'][0]
                coord1 = model_input['mask_block_coords1'][0]
                coord2 = model_input['mask_block_coords2'][0]
                pred_2img[:,coord0,coord1,coord2,:] = pred_imgline.permute(0,2,1).cpu()
                
                preds.append(pred_2img.numpy())
                if ii == coord_dataset.length-1 or dpSub != idxdict[ii+1][2]:
                    brain_tensor,_ = qtlib.block2brain(np.concatenate(preds,0),coord_dataset.img_datasets[idxdict[ii][0]].ind_block,coord_dataset.img_datasets[idxdict[ii][0]].mask)
                    brain_tensor[:,:,:,-1] = brain_tensor[:,:,:,-1]*b0rescale

                    if direapply:
                        print('change the save dire in the code by hand')
                        # qtlib.save_nii(...)
                        # qtlib.save_nii(...)
                        pass
                    else:
                        if bestval=='bestval':
                            if inf==0:
                                outputtensor = brain_tensor
                            else:
                                outputtensor = outputtensor + brain_tensor
                        elif bestval=='ckbest':
                            if inf==0:
                                outputtensor = brain_tensor
                            else:
                                outputtensor = outputtensor + brain_tensor
                        elif bestval in ['10']:
                            if inf==0:
                                outputtensor = brain_tensor
                            else:
                                outputtensor = outputtensor + brain_tensor
                        elif bestval in ['0','1']:
                            if inf==0:
                                outputtensor = brain_tensor
                            else:
                                outputtensor = outputtensor + brain_tensor
                        else:
                            assert(0)
                    preds = []
                    if ii != coord_dataset.length-1:
                        dpSub = idxdict[ii+1][2]
            pred_curriter.append(brain_tensor)
        
        if bestval=='bestval':
            qtlib.save_nii(os.path.join(dpSub,eval_dir,f'testpred_bestAvgval.nii.gz'),outputtensor/numMC,fpref)
        elif bestval=='ckbest':
            qtlib.save_nii(os.path.join(dpSub,eval_dir,f'{opt.subj}_pred_ckAvgbest.nii.gz'),outputtensor/numMC,fpref)
        elif bestval in ['10']:
            qtlib.save_nii(os.path.join(dpSub,eval_dir,f'{opt.subj}_pred_epoch{int(numiter):06d}.nii.gz'),outputtensor/numMC,fpref)
        elif bestval in ['1']:
            qtlib.save_nii(os.path.join(dpSub,eval_dir,f'{opt.subj}_pred_epoch{int(curr_iter):06d}.nii.gz'),outputtensor/numMC,fpref)
        elif bestval in ['0']:
            qtlib.save_nii(os.path.join(dpSub,eval_dir,f'{opt.subj}_pred_epoch{int(curr_iter):06d}.nii.gz'),outputtensor/numMC,fpref)
        else:
            qtlib.save_nii(os.path.join(dpSub,eval_dir,f'{opt.subj}_testpred.nii.gz'),outputtensor/numMC,fpref)
            pass
        if bestval in ['bestval', 'ckbest','10']:
            break  



if __name__ == '__main__':
    main()
