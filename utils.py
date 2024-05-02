import numpy as np
import torch
import os
from torchvision.utils import make_grid
from tqdm import tqdm
import modules
import loss_functions
from functools import partial
import math
torch.cuda.set_device(int(os.environ["CUDA_USING"]))
def cond_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def dict2cuda(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value.cuda(int(os.environ["CUDA_USING"]))})
        else:
            tmp.update({key: value})
    return tmp


def dict2cpu(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value.cpu()})
        elif isinstance(value, dict):
            tmp.update({key: dict2cpu(value)})
        else:
            tmp.update({key: value})
    return tmp


def process_batch_in_chunks(in_dict, model, max_chunk_size=1024, progress=None):
    in_chunked = []
    for key in in_dict:
        chunks = torch.split(in_dict[key], max_chunk_size, dim=1)
        in_chunked.append(chunks)

    list_chunked_batched_in = \
        [{k: v for k, v in zip(in_dict.keys(), curr_chunks)} for curr_chunks in zip(*in_chunked)]
    del in_chunked

    list_chunked_batched_out_out = {}
    list_chunked_batched_out_in = {}
    for chunk_batched_in in tqdm(list_chunked_batched_in):
        chunk_batched_in = {k: v.cuda(int(os.environ["CUDA_USING"])) for k, v in chunk_batched_in.items()}
        tmp = model(chunk_batched_in)
        tmp = dict2cpu(tmp)

        for key in tmp['model_out']:
            if tmp['model_out'][key] is None:
                continue

            out_ = tmp['model_out'][key].detach().clone().requires_grad_(False)
            list_chunked_batched_out_out.setdefault(key, []).append(out_)

        for key in tmp['model_in']:
            if tmp['model_in'][key] is None:
                continue

            in_ = tmp['model_in'][key].detach().clone().requires_grad_(False)
            list_chunked_batched_out_in.setdefault(key, []).append(in_)

        del tmp, chunk_batched_in

    # Reassemble the output chunks in a batch
    batched_out = {}
    for key in list_chunked_batched_out_out:
        batched_out_lin = torch.cat(list_chunked_batched_out_out[key], dim=1)
        batched_out[key] = batched_out_lin

    batched_in = {}
    for key in list_chunked_batched_out_in:
        batched_in_lin = torch.cat(list_chunked_batched_out_in[key], dim=1)
        batched_in[key] = batched_in_lin

    return {'model_in': batched_in, 'model_out': batched_out}


def subsample_dict(in_dict, num_views, multiscale=False):
    if multiscale:
        out = {}
        for k, v in in_dict.items():
            if v.shape[0] == in_dict['octant_coords'].shape[0]:
                # this is arranged by blocks
                out.update({k: v[0:num_views[0]]})
            else:
                # arranged by rays
                out.update({k: v[0:num_views[1]]})
    else:
        out = {key: value[0:num_views, ...] for key, value in in_dict.items()}

    return out

def get_model(opt,in_features, out_features):
    model = modules.FastCNNBaseNetwork3D( in_features=in_features+opt.useT1+opt.useT2,
                                out_features=out_features,
                                num_hidden_layers=opt.hidden_layers,
                                num_encoding_layers=opt.num_encoding_layers,
                                hidden_features=opt.hidden_features, # 64 for start code
                                encoding_features=opt.encoding_features,
                                mode='pe',conv1 = opt.conv1,paranetwork=opt.paranetwork,convnetwork=opt.convnetwork,DropRate = opt.DropRate,DropoutType=opt.DropoutType,ngroups=opt.ngroups,
                                opt=opt)

    return model
def noutfeas(opt):
    if opt.diffusionmodel == 'tensor':
        out_features = 7
    else:
        assert(0)
    return out_features
def DeterLoss(opt,coord_dataset):

    lossclass = loss_functions.FastgrdenlossMLP3D(opt)
    loss_fn = partial(lossclass.image_mse_diffusion, # loss_functions.image_mse,
                    tiling_every=opt.steps_til_tiling,
                    dataset=coord_dataset,
                    model_type=opt.model_type)
    return loss_fn

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def NoneEdge(block_coords,opt):
    filterindexes = []
    for coords in block_coords:
        edgeindex = np.concatenate([np.where(coords[0]==0)[0],np.where(coords[0]==opt.sz_block-1)[0],
                                    np.where(coords[1]==0)[0],np.where(coords[1]==opt.sz_block-1)[0],
                                    np.where(coords[2]==0)[0],np.where(coords[2]==opt.sz_block-1)[0]],0)
        edgeindex = np.unique(edgeindex)
        allindex = np.arange(0,len(coords[0]))
        filterindex = allindex[~np.isin(allindex,edgeindex)]
        filterindexes.append(filterindex)
        # print(filterindex)
        # print(np.sum(coords[0][filterindex]) )
        # print(np.all(filterindex[:-1] <= filterindex[1:])) # 自动升序排列的
    return filterindexes


def write_image_patch_multiscale_summary(image_resolution, patch_size, dataset, model, model_input, gt,
                                         model_output, writer, total_steps, prefix='train_',
                                         model_type='multiscale', skip=False):
    if skip:
        return

    # uniformly sample the image
    dataset.toggle_eval()
    model_input, gt = dataset[0]
    dataset.toggle_eval()

    # convert to cuda and add batch dimension
    tmp = {}
    for key, value in model_input.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value[None, ...].cpu()})
        else:
            tmp.update({key: value})
    model_input = tmp

    tmp = {}
    for key, value in gt.items():
        if isinstance(value, torch.Tensor):
            tmp.update({key: value[None, ...].cpu()})
        else:
            tmp.update({key: value})
    gt = tmp

    # run the model on uniform samples
    # n_channels = gt['img'].shape[-1]
    n_channels = 6
    pred_img = process_batch_in_chunks(model_input, model)['model_out']['output']

    # get pixel idx for each coordinate
    coords = model_input['fine_abs_coords'].detach().cpu().numpy()
    pixel_idx = np.zeros_like(coords).astype(np.int32)
    pixel_idx[..., 0] = np.round((coords[..., 0] + 1.)/2. * (dataset.sidelength[0]-1)).astype(np.int32)
    pixel_idx[..., 1] = np.round((coords[..., 1] + 1.)/2. * (dataset.sidelength[1]-1)).astype(np.int32)
    pixel_idx = pixel_idx.reshape(-1, 2)

    # get pixel idx for each coordinate in frozen patches
    frozen_coords, frozen_values = dataset.get_frozen_patches()
    if frozen_coords is not None:
        frozen_coords = frozen_coords.detach().cpu().numpy()
        frozen_pixel_idx = np.zeros_like(frozen_coords).astype(np.int32)
        frozen_pixel_idx[..., 0] = np.round((frozen_coords[..., 0] + 1.) / 2. * (dataset.sidelength[0] - 1)).astype(np.int32)
        frozen_pixel_idx[..., 1] = np.round((frozen_coords[..., 1] + 1.) / 2. * (dataset.sidelength[1] - 1)).astype(np.int32)
        frozen_pixel_idx = frozen_pixel_idx.reshape(-1, 2)

    # init a new reconstructed image
    display_pred = np.zeros((*dataset.sidelength, n_channels))

    # assign predicted image values into a new array
    # need to use numpy since it supports index assignment
    pred_img = pred_img.reshape(-1, n_channels).detach().cpu().numpy()
    display_pred[[pixel_idx[:, 0]], [pixel_idx[:, 1]]] = pred_img

    # assign frozen image values into the array too
    if frozen_coords is not None:
        frozen_values = frozen_values.reshape(-1, n_channels).detach().cpu().numpy()
        display_pred[[frozen_pixel_idx[:, 0]], [frozen_pixel_idx[:, 1]]] = frozen_values

    # show reconstructed img
    display_pred = torch.tensor(display_pred)[None, ...]
    display_pred = display_pred.permute(0, 3, 1, 2)

    gt_img = gt['img'].reshape(-1, n_channels).detach().cpu().numpy()
    display_gt = np.zeros((*dataset.sidelength, n_channels))
    display_gt[[pixel_idx[:, 0]], [pixel_idx[:, 1]]] = gt_img
    display_gt = torch.tensor(display_gt)[None, ...]
    display_gt = display_gt.permute(0, 3, 1, 2)

    fig = dataset.quadtree.draw()
    writer.add_figure(prefix + 'tiling', fig, global_step=total_steps)

    if 'img' in gt:
        output_vs_gt = torch.cat((display_gt, display_pred), dim=0)
        writer.add_image(prefix + 'gt_vs_pred', make_grid(output_vs_gt, scale_each=False, normalize=True),
                         global_step=total_steps)
        write_psnr(display_pred, display_gt, writer, total_steps, prefix+'img_')

