'''Implements a generic training loop.
'''

import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
from torch import nn
from torch.optim import AdamW
import time
import random
import pickle
import qtlib
import pdb
from torch_ema import ExponentialMovingAverage
torch.cuda.set_device(int(os.environ["CUDA_USING"]))
def trainMCvalidationLight(model, train_dataloader, epochs, lr, steps_til_summary, epochs_til_checkpoint, model_dir,
          loss_fn, pruning_fn, summary_fn, double_precision=False, clip_grad=False,optimizer = 'Adam',
          loss_schedules=None, resume_checkpoint={}, objs_to_save={}, epochs_til_pruning=4,subjpath=None,opt=None,coord_dataset = None):
    # reset the batchnorm
    def reset_batchnorm(m): 
        if type(m) == nn.BatchNorm3d: m.reset_parameters()
        if type(m) == nn.BatchNorm1d: m.reset_parameters()
    # model.apply(reset_batchnorm)'
    if optimizer == 'Adam':
        optim = torch.optim.Adam(lr=lr, params=model.parameters())
    elif optimizer == 'AdamW':
        print('using AdamW!!!')
        optim = AdamW(lr = lr,params = model.parameters(),weight_decay=0.05,betas=(0.9,0.999))
    start=time.time()
    # load optimizer if supplied
    if 'optimizer_state_dict' in resume_checkpoint:
        optim.load_state_dict(resume_checkpoint['optimizer_state_dict'])
    for g in optim.param_groups:
        g['lr'] = lr
    
    if opt.representation == 1 and opt.trainingstrate==1:
        optim = torch.optim.Adam(lr = lr, params = model.parameters(), betas=(0.9, 0.99), eps=1e-15)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda iter: 0.1 ** min(iter / (len(train_dataloader) * epochs), 1))
        ema = ExponentialMovingAverage(model.parameters(), decay=0.95)

    os.makedirs(model_dir, exist_ok=True)

    summaries_dir = os.path.join(model_dir, 'summaries')
    utils.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)
    total_steps = 0
    if 'total_steps' in resume_checkpoint:
        total_steps = resume_checkpoint['total_steps']

    start_epoch = 0
    if 'epoch' in resume_checkpoint:
        start_epoch = resume_checkpoint['epoch']

    minvalloss = 100000
    minvalloss2 = 100000
    epochloss = 100000
    minepochloss = 100000
    earlystopdone = 0
    earlycount = 0
    winsize = 10
    minwinval = 10000000
    minwinvalindex = 0
    valid_loss = minvalloss
    start=time.time()
    if len(train_dataloader) * epochs>1000: # add for FS
        tmpepochs_til_checkpoint = epochs_til_checkpoint # add for FS
        epochs_til_checkpoint = 10 # add for FS
    torch.save(model.state_dict(), os.path.join(checkpoints_dir, '1best_val_model_000000.pth'))
    bestmodel = None
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        pbar.update(total_steps)
        train_losses = []
        valid_losses = []
        es_valid_losses = []
        es_valid_losses_markinepoch = []
        es_valid_losses_markinepoch_win = []  
        for epoch in range(start_epoch, epochs):
            epochstarttime = time.time()
            if opt.MCchannel:
                usechannel = coord_dataset.bvalnotzero
                print(usechannel)
            if valid_loss<minvalloss2:
                minvalloss2 = valid_loss
                torch.save(model.state_dict(),
                os.path.join(checkpoints_dir, 'ck0_val_model.pth'))
            if epoch>=100: # add for FS
                epochs_til_checkpoint = tmpepochs_til_checkpoint # add for FS
            if not epoch % epochs_til_checkpoint and epoch:
                updateornot = int(valid_loss<minvalloss)
                dpbest2now = os.path.join(checkpoints_dir, 'ck0_val_model.pth')
                dpnow = os.path.join(checkpoints_dir, f'{updateornot:1d}best_val_model_{total_steps:06d}.pth')
                
                os.system(f"cp {dpbest2now} {dpnow}")
                
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_%06d.txt' % total_steps),
                           np.array(train_losses))
                np.savetxt(os.path.join(checkpoints_dir, 'es_valid_losses_%06d.txt' % total_steps),
                           np.array(es_valid_losses))
            if not (epoch-winsize//2) % epochs_til_checkpoint and epoch>winsize:  
                es_valid_losses_markinepoch_win.append( np.average(es_valid_losses_markinepoch[epoch-winsize:epoch])  )
                if es_valid_losses_markinepoch_win[-1]<minwinval:
                    minwinval = es_valid_losses_markinepoch_win[-1]
                    minwinvalindex = epoch-winsize//2
                if epoch >= 10*winsize and not earlystopdone:
                    if (es_valid_losses_markinepoch_win[-2]-es_valid_losses_markinepoch_win[-1])/es_valid_losses_markinepoch_win[-1]<0.0001:
                        earlycount = earlycount + 1
                        if earlycount > 2:
                            dpearlystop = os.path.join(checkpoints_dir,f"ES_{(epoch-winsize//2)}")
                            if not os.path.exists(dpearlystop):
                                os.mkdir(dpearlystop)
                            
                            dpearlystop = os.path.join(checkpoints_dir,f"ES_min_{minwinvalindex}")
                            if not os.path.exists(dpearlystop):
                                os.mkdir(dpearlystop)
                            earlystopdone = 1
                        
                    else:
                        earlycount = 0
            if epoch in [10,50,100,300]:
                dpbest2now = os.path.join(checkpoints_dir, 'ck0_val_model.pth')
                dpnow = os.path.join(checkpoints_dir, f'epoch{epoch:06d}.pth')
                os.system(f"cp {dpbest2now} {dpnow}")
            if not (epoch + 1) % epochs_til_pruning:
                retile = False
            else:
                retile = True
            count = 0
            length = len(train_dataloader)
            if opt.cv:
                splitnum = int(length*0.8)
            else:
                splitnum = int(length)
            epochlosslist = []
            validlosslist = []
            esvalidloss_epochs = []
            stependtime = time.time()
            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
                tmp = {}
                for key, value in model_input.items():
                    if isinstance(value, torch.Tensor):
                        tmp.update({key: value.cuda(int(os.environ["CUDA_USING"]))})
                        # print(key,time.time()-start_time)
                    else:
                        tmp.update({key: value})
                model_input = tmp

                tmp = {}
                for key, value in gt.items():
                    if isinstance(value, torch.Tensor):
                        tmp.update({key: value.cuda(int(os.environ["CUDA_USING"]))})
                        # print('gt',key,time.time()-start_time)
                    else:
                        tmp.update({key: value})
                gt = tmp
                # '''
                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}
                if step<=splitnum:
                    if opt.sz_block_mode in ['min32']:
                        if len(gt['NoneEdgeIndex'][0])==0:
                            continue
                    model_output = model(model_input)

                    losses = loss_fn(model_output, gt, total_steps, retile=retile)

                    train_loss = 0.
                    es_valid_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        if loss_schedules is not None and loss_name in loss_schedules:
                            writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                            single_loss *= loss_schedules[loss_name](total_steps)

                        writer.add_scalar(loss_name, single_loss, total_steps)
                        if loss_name == 'img_loss':
                            train_loss += single_loss
                        elif loss_name == 'valid_loss':
                            es_valid_loss += single_loss

                    train_losses.append(train_loss.item())
                    if opt.EarlyStopping or opt.MCchannel:
                        es_valid_losses.append(es_valid_loss.item())
                        esvalidloss_epochs.append(es_valid_loss.item())
                    else:
                        es_valid_losses.append(es_valid_loss)
                        esvalidloss_epochs.append(es_valid_loss)

                    writer.add_scalar("total_train_loss", train_loss, total_steps)
                    writer.add_scalar("total_es_valid_loss", es_valid_loss, total_steps)
                    optim.zero_grad()
                    train_loss.backward()

                    if clip_grad:
                        if isinstance(clip_grad, bool):
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                    optim.step()
                    pbar.update(1)
                    epochlosslist.append(train_loss.detach().item())
                    total_steps += 1
                    if opt.representation == 1 and opt.trainingstrate==1:
                        lr_scheduler.step()
                else:
                    with torch.no_grad():
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt, total_steps, retile=retile)

                        valid_loss = 0.
                        es_valid_loss = 0.
                        for loss_name, loss in losses.items():
                            single_loss = loss.mean()

                            if loss_schedules is not None and loss_name in loss_schedules:
                                writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                                single_loss *= loss_schedules[loss_name](total_steps)

                            writer.add_scalar(loss_name, single_loss, total_steps)
                            valid_loss += single_loss
                            if loss_name == 'img_loss':
                                train_loss += single_loss
                            elif loss_name == 'valid_loss':
                                es_valid_loss += single_loss

                        valid_losses.append(valid_loss.item())
                        validlosslist.append(valid_loss.item())
                        epochlosslist.append(valid_loss.item())
                        if opt.EarlyStopping or opt.MCchannel:
                            es_valid_losses.append(es_valid_loss.item())
                            esvalidloss_epochs.append(es_valid_loss.item())
                        else:
                            es_valid_losses.append(es_valid_loss)
                            esvalidloss_epochs.append(es_valid_loss)
                        writer.add_scalar("total_valid_loss", valid_loss, total_steps)
                        writer.add_scalar("total_es_valid_loss", es_valid_loss, total_steps)
                if opt.representation == 1 and opt.trainingstrate==1:
                    ema.update()
            train_loss = np.average(epochlosslist)
            if opt.cv:
                valid_loss = np.average(validlosslist)
            else:
                valid_loss = np.average(epochlosslist)
            esvalidloss_epoch = np.average(esvalidloss_epochs)
            es_valid_losses_markinepoch.append(esvalidloss_epoch)
            # if opt.earlystop:


            tqdm.write("Epoch %d, Total loss %0.6f, Valid loss %0.6f, es Valid loss %0.6f, iteration time %0.6f, epoch time %0.6f" % (epoch, train_loss, valid_loss, esvalidloss_epoch, time.time() - start_time, time.time()-epochstarttime))
        if 0:
            state = {
                'epoch': epochs,
                'global_step': total_steps,
             }
            state['optimizer'] = optim.state_dict()
            state['lr_scheduler'] = lr_scheduler.state_dict()
            state['ema'] = ema.state_dict()
            state['model'] = model.state_dict()
            torch.save(state, os.path.join(checkpoints_dir, 'model_final_%06d.pth' % total_steps))
        else:
            torch.save(model.state_dict(),
                    os.path.join(checkpoints_dir, 'model_final_%06d.pth' % total_steps))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final_%06d.txt' % total_steps),
                   np.array(train_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'valid_losses_final_%06d.txt' % total_steps),
                np.array(valid_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'es_valid_losses_final_%06d.txt' % total_steps),
                np.array(es_valid_losses))
        np.savetxt(os.path.join(checkpoints_dir, 'es_valid_losses_inepoch_final_%06d.txt' % total_steps),
                np.array(es_valid_losses_markinepoch))
