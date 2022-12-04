import sys
import torch
import torch.nn.functional as F
from packaging import version
import numpy as np
from collections import OrderedDict

import torch.nn as nn

class ContentLoss(nn.Module):
    
    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input, alpha=1.):
        self.loss = F.mse_loss(alpha*input, self.target)
        return input    

def gram_matrix_ori(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):
    
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix_ori(target_feature).detach()

    def forward(self, input):
        G = gram_matrix_ori(input)
        self.loss = F.mse_loss(G, self.target)
        return input
