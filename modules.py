import torch
from torch import batch_norm, nn
import numpy as np
import math
from functools import partial
from matplotlib import pyplot as plt
import os
from modulesunit import Sine, sine_init, first_layer_sine_init, init_weights_normal, GaussianDropout, ResidualBlock
torch.cuda.set_device(int(os.environ["CUDA_USING"]))



class FastCNNBlock3DConv2PartDropout(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,encoding_features,
                 num_encoding_layers=1,outermost_linear=False, nonlinearity='relu', weight_init=None, w0=30,FOVin = 1,DropRate=0.1,DropoutType="MC"):
        super().__init__()

        self.first_layer_init = None

        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        nls_and_inits = {'relu': (nn.ReLU(inplace=True), init_weights_normal, None)}

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net_extrafea = []
        self.net_process = []

        if num_encoding_layers == 1:
            self.net_extrafea.append(nn.Sequential(
                nn.Conv3d(in_features,encoding_features,kernel_size=FOVin,padding=FOVin//2,padding_mode='zeros'),nl # nn.Linear(in_features, hidden_features), nl
            ))
        elif num_encoding_layers >=2:
            self.net_extrafea.append(nn.Sequential(
                nn.Conv3d(in_features,encoding_features,kernel_size=FOVin,padding=FOVin//2,padding_mode='zeros'),nl # nn.Linear(in_features, hidden_features), nl
            ))
            for i in range(num_encoding_layers-2):
                self.net_extrafea.append(nn.Sequential(
                    nn.Conv3d(encoding_features,encoding_features,kernel_size=FOVin,padding=FOVin//2,padding_mode='zeros'),nl # nn.Linear(in_features, hidden_features), nl
                ))
            self.net_extrafea.append(nn.Sequential(
                    nn.Conv3d(encoding_features,encoding_features,kernel_size=FOVin,padding=FOVin//2,padding_mode='zeros'),nl # nn.Linear(in_features, hidden_features), nl
                ))
        elif num_encoding_layers == 0:
            if DropoutType=="MC":
                self.net_process.append(nn.Sequential(
                    nn.Conv1d(in_features,hidden_features,kernel_size=1,padding=0,padding_mode='zeros'),nl, # nn.Linear(hidden_features, hidden_features), nl
                    nn.Dropout(DropRate)
                ))
                encoding_features = hidden_features
                num_encoding_layers = 1
            else:
                assert(0)
        if DropoutType=="MC":
            self.net_process.append(nn.Sequential(
                nn.Conv1d(encoding_features,hidden_features,kernel_size=1,padding=0,padding_mode='zeros'),nl, # nn.Linear(hidden_features, hidden_features), nl
                nn.Dropout(DropRate)
            ))
            for i in range(num_hidden_layers-num_encoding_layers-1):
                self.net_process.append(nn.Sequential(
                nn.Conv1d(hidden_features,hidden_features,kernel_size=1,padding=0,padding_mode='zeros'),nl, # nn.Linear(hidden_features, hidden_features), nl
                nn.Dropout(DropRate)
            ))
        elif DropoutType=='Gau':
            self.net_process.append(nn.Sequential(
                nn.Conv1d(encoding_features,hidden_features,kernel_size=1,padding=0,padding_mode='zeros'),nl, # nn.Linear(hidden_features, hidden_features), nl
                GaussianDropout(DropRate)
            ))
            for i in range(num_hidden_layers-num_encoding_layers-1):
                self.net_process.append(nn.Sequential(
                nn.Conv1d(hidden_features,hidden_features,kernel_size=1,padding=0,padding_mode='zeros'),nl, # nn.Linear(hidden_features, hidden_features), nl
                GaussianDropout(DropRate)
            ))

        if outermost_linear:
            self.net_process.append(nn.Sequential(
                nn.Conv1d(hidden_features,64,kernel_size=1,padding=0,padding_mode='zeros'),
                nn.Conv1d(64,out_features,kernel_size=1,padding=0,padding_mode='zeros')
            ))
        else:
            self.net_process.append(nn.Sequential(
                nn.Conv1d(hidden_features,64,kernel_size=1,padding=0,padding_mode='zeros'),nl,
                nn.Conv1d(64,out_features,kernel_size=1,padding=0,padding_mode='zeros'),nl
            ))

        self.net_extrafea = nn.Sequential(*self.net_extrafea)
        self.net_process = nn.Sequential(*self.net_process)

        if self.weight_init is not None:
            self.net_extrafea.apply(self.weight_init)
            self.net_process.apply(self.weight_init)
        if first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.net_extrafea[0].apply(first_layer_init)

    def forward(self, inputs):
        features = self.net_extrafea(inputs['img_block']) # plt.imshow(inputs['img_block'][0,16,:,:,50].detach().cpu().numpy(),'gray')
        mask_block_coords0 = inputs['mask_block_coords0']
        mask_block_coords1 = inputs['mask_block_coords1']
        mask_block_coords2 = inputs['mask_block_coords2']
        output = self.net_process(features[:,:,mask_block_coords0[0],mask_block_coords1[0],mask_block_coords2[0]])
        return output

class FastCNNBaseNetwork3D(nn.Module):
    def __init__(self, in_features=3, out_features=1, feature_grid_size=(8, 8, 8),
                 hidden_features=256,encoding_features=0, num_hidden_layers=3,num_encoding_layers=2, patch_size=8,
                 code_dim=8, use_pe=True, num_encoding_functions=6, conv1 = 3,paranetwork=False,convnetwork=0,DropRate=0.1,DropoutType='MC',opt=None,**kwargs):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.feature_grid_size = feature_grid_size
        self.patch_size = patch_size
        self.conv1 = conv1
        self.convnetwork = convnetwork
        self.opt = opt
        # note! num_encoding_layers + numnum_hidden_layers = 
        if encoding_features == 0:
            encoding_features = hidden_features

        if convnetwork==3: # using the dropout
            self.img2tensor_net = FastCNNBlock3DConv2PartDropout(in_features=in_features, out_features=out_features, encoding_features=encoding_features,
                                            num_hidden_layers=num_hidden_layers,num_encoding_layers=num_encoding_layers, hidden_features=hidden_features,
                                            outermost_linear=True, nonlinearity='relu',FOVin = self.conv1,DropRate = DropRate,DropoutType=DropoutType)


    def forward(self, model_input):

        # Enables us to compute gradients w.r.t. coordinates
        # coords = model_input['coords'].clone().detach().requires_grad_(True)
        # fine_coords = model_input['fine_rel_coords'].clone().detach().requires_grad_(True)
       
        img_block = model_input['img_block'].clone().detach().requires_grad_(True)
        mask_block_coords0 = model_input['mask_block_coords0']
        mask_block_coords1 = model_input['mask_block_coords1']
        mask_block_coords2 = model_input['mask_block_coords2']
        indict = {'img_block':img_block,'mask_block_coords0':mask_block_coords0,'mask_block_coords1':mask_block_coords1,'mask_block_coords2':mask_block_coords2,'bvec':model_input['bvec']}
        
        out_block = self.img2tensor_net( indict )
        return {'model_in': model_input,
                'model_out': {'output': out_block, 'codes': None}}