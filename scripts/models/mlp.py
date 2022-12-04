import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.utils import *
import numpy as np
from typing import Tuple

class MLPWithInputSkips(nn.Module):
    def __init__(self, args,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips: Tuple[int] = ()):
        super().__init__()

        print('*hidden_dim :',hidden_dim)

        self.args = args
        self.mode = 'skip'

        layers = []
        for layeri in range(n_layers) : #8
            if layeri ==0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                if self.mode == 'skip':
                    dimin = hidden_dim + skip_dim
                else :
                    dimin = hidden_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = nn.Linear(dimin, dimout)
            _xavier_init(linear)
            layers.append(nn.Sequential(linear, torch.nn.ReLU(True)))
        
        self.mlp = nn.ModuleList(layers)
                
        if self.mode == 'skip':
            self._input_skips = set(input_skips)
            print('*Input Skip Activated')
        else :
            self._input_skips = set()

    def forward(self, x: torch.Tensor, z: torch.Tensor) : # z=x just for input skip
        y = x
        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)
            y = layer(y)
        return y



class LatentMLP(nn.Module):
    def __init__(
        self,
        args,
        n_harmonic_functions_xyz: int = 6,
        n_harmonic_functions_dir: int = 4,
        n_hidden_neurons_xyz: int = 256,
        n_hidden_neurons_dir: int = 128,
        n_layers_xyz: int = 8,
        append_xyz: Tuple[int] = (5,),
        use_multiple_streams: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(n_harmonic_functions_xyz).to(args.device)  #6
        embedding_dim_xyz = args.latent_size + 26 #n_harmonic_functions_xyz * 2 * 3 + 3

        # print('embedding_dim_xyz:',embedding_dim_xyz) torch.Size([65536, 282 = 64 + 26])

        self.mlp = MLPWithInputSkips(
            args,
            args.depth,                   #8 depth n_layers_xyz
            embedding_dim_xyz,              # 26 harmonic embedding_dim_xyz
            args.width,           #256 n_hidden_neurons_xyz
            embedding_dim_xyz,              # 26 harmonic embedding_dim_xyz
            args.width,           #256 n_hidden_neurons_xyz
            input_skips=append_xyz,         #(5,) input_skips=append_xyz
            ).to(args.device)
        self.intermediate_linear = nn.Linear(args.width, args.width).to(args.device)
        _xavier_init(self.intermediate_linear)

        self.intermediate_last_linear = nn.Linear(args.width, n_hidden_neurons_dir).to(args.device)
        _xavier_init(self.intermediate_last_linear)

        self.color_layer = nn.Sequential(
            # LinearWithRepeat(
            #     n_hidden_neurons_xyz + embedding_dim_dir, n_hidden_neurons_dir
            # ),
            torch.nn.Linear(n_hidden_neurons_dir, 3),
            torch.nn.Sigmoid(),
        ).to(args.device)
    
    def forward(self, coord, z):
        # print('coord :', coord.size()) coord : torch.Size([65536, 2])
        embed_coord = self.harmonic_embedding_xyz(coord)
        cat_coord = torch.cat((embed_coord, z), dim=1)
        # print('cat_coord :', cat_coord.size())   #cat_coord : torch.Size([65536, 282])
        output = self.mlp(cat_coord,cat_coord)
        output = self.intermediate_last_linear(output) #256 to 128
        output = self.color_layer(output) #128 to 3
        # print('output :', output.size())output : torch.Size([65536, 3])

        return output

class HarmonicEmbedding(nn.Module):
    def __init__(
        self,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ):
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input

    def forward(self, x):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)
        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)

def get_network(args, scale, img_size=None):
    network = LatentMLP(args)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.out_lr, betas=(args.beta1, args.beta2))
    if img_size is None:
        coord = get_mgrid(args.img_size)
    else:
        coord = get_mgrid(img_size)
        
    coord = coord.type(torch.cuda.FloatTensor).to(args.device).detach()

    print('coord :', coord.size()) 

    return network, optimizer, coord

def get_network_eval(args, img_size=None):
    
    network = LatentMLP(args)
    
    if img_size is None:
        coord = get_mgrid(args.img_size)
    else:
        coord = get_mgrid(img_size)
    coord = coord.type(torch.cuda.FloatTensor).to(args.device).detach()
    print('coord :', coord.size()) 
    return network, coord


def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def _xavier_init(linear):
    """
    Performs the Xavier weight initialization of the linear layer `linear`.
    """
    torch.nn.init.xavier_uniform_(linear.weight.data)