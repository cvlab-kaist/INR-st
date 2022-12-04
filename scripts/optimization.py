import random
import os
import cv2
import numpy as np
import torch
from PIL import Image
import argparse
import gc
import random
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as mods
import torchvision.transforms as transforms

from models.mlp import *
from models.utils import *

parser = argparse.ArgumentParser(description='Code to optimize')
parser.add_argument('--kappa',        help="exponential value",  default=1.0, type=float)

parser.add_argument('--device',                  help='cuda | cuda:0 | cpu',                   default="cuda", type=str)
parser.add_argument('--device_num',              help='which GPUs to use',                     default="0",  type=str)

parser.add_argument('--iter',         help="number of iteration for optimization", default=20000, type=int)
parser.add_argument('--img_size',     help="size of input image",                  default=256, type=int)
parser.add_argument('--content_wt',        help="weight of content", default=1, type=float)
parser.add_argument('--style_wt',        help="weight of style",default=100000, type=float)
parser.add_argument('--beta1',        help="optimizer parameter",                  default=0.5, type=float)
parser.add_argument('--beta2',        help="optimizer parameter",                  default=0.99, type=float)
parser.add_argument('--out_lr',           help="learning rate",                        default=1e-3, type=float)

"""Test-Time Optimization Settings"""
parser.add_argument('--content_path', required=True, type=str, help='path to content image')
parser.add_argument('--style_path', required=True, type=str, help='path to style image')
parser.add_argument('--save_dir', default='./checkpoints',help='Directory to save the model')
parser.add_argument('--start_iter', type=int, default=0)
parser.add_argument('--type', default='LatentMLP', type=str, help='type of model')
parser.add_argument('--layer',default='shallow_relu', type=str, help='deep or shallow')
parser.add_argument('--latent_size',default=16, type=int, help='size of latent vector')
parser.add_argument('--depth',default=8, type=int, help='model depth')
parser.add_argument('--width',default=256, type=int, help='feature size')

args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
GPU_list = args.device_num
len_GPU = len( GPU_list.split(","))
print("*len_GPU: ", len_GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_list

random_seed = 1006
os.environ['PYTHONHASHSEED'] = str(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(args.device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(args.device)

#  Load content and style
def img_load(path):
    img = cv2.imread(path)[::,::,::-1] # BGR to RGB, [0-255]
    return img

def toPIL(img):
    img_type = str(type(img))
    if 'numpy' in img_type:
        img = Image.fromarray(img)
    elif 'torch' in img_type:
        img = transforms.ToPILImage()(img).convert("RGB")
    return img

content_path =   args.content_path
style_path   =   args.style_path

c_img = os.path.basename(content_path) 
s_img = os.path.basename(style_path)  
c_name = c_img.split('.')[0] 
s_name = s_img.split('.')[0] 


def get_input_optimizer(input_img):
    optimizer = torch.optim.LBFGS([input_img.requires_grad_()])
    return optimizer

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
      
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std

# desired depth layers to compute style/content losses :
if args.layer == 'shallow':
    print(f"{args.layer}")
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
elif args.layer == 'deep':
    print(f"{args.layer}")
    content_layers_default = ['conv_9']
    style_layers_default = ['conv_1', 'conv_3', 'conv_5', 'conv_9', 'conv_13']
elif args.layer == 'deep_relu':
    print(f"{args.layer}")
    content_layers_default = ['relu_9']
    style_layers_default = ['relu_1', 'relu_3', 'relu_5', 'relu_9', 'relu_13']
elif args.layer == 'shallow_relu':
    print(f"{args.layer}")
    content_layers_default = ['relu_4']
    style_layers_default = ['relu_1', 'relu_2', 'relu_3', 'relu_4', 'relu_5']

def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(args.device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0  # increment every time we see a conv
    j = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            j+=1
            name = 'relu_{}'.format(j)
       
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        if isinstance(layer, nn.MaxPool2d):
            model.add_module(name, nn.AvgPool2d(2))
        else:
            model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def mlp_forward(args, network, coords, z):
    #LatentMLP
    output = network(coords, z.repeat(args.img_size**2, 1))
    output = output.permute(1,0)
    output = output.view(-1,args.img_size,args.img_size)
    output = torch.unsqueeze( output, dim=0 )
    return output.type(torch.cuda.FloatTensor).to(args.device)

def reweighting(alpha, k=1):

    return (-1* (1+(alpha-1))**k) * ( torch.log(-alpha+1) )

def run_style_transfer_inr(cnn,network,optimizer, normalization_mean, normalization_std,
                       content_img, style_img, coords ,content_original,style_original,latent_c,latent_s, num_steps=args.iter,
                       style_weight=args.style_wt, content_weight=args.content_wt):
    print('Building the style transfer model..')

    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    model.requires_grad_(False) # VGG Fix

    result = None
    start = time.time()

    prog_bar = tqdm(range(args.iter- args.start_iter))
   
    z_c = latent_c
    z_s = latent_s

    for i in prog_bar:
        total_step = i + args.start_iter
        network.train()

        alpha_1 = np.random.rand(1) 
        alpha_1 = torch.cuda.FloatTensor(alpha_1).to(args.device).detach()
      
        style_score_1   = torch.tensor(0., device=args.device)
        content_score_1 = torch.tensor(0., device=args.device)
   
        z_1 = alpha_1*z_c + (1-alpha_1)*z_s #interpolated latent vector

        input_img_1 = mlp_forward(args, network, coords, z_1)
        input_img_1.requires_grad_(True)

        
        model(input_img_1) #VGG Loss
        for i, sl in enumerate(style_losses):
            style_score_1 += sl.loss
        for j, cl in enumerate(content_losses):
            content_score_1 += cl.loss

        weighted_style_score_1 = style_weight*(style_score_1)
        weighted_content_score_1 = content_weight*(content_score_1)
        reweighted_content_score_1 = reweighting(alpha_1,args.kappa)*weighted_content_score_1
        reweighted_style_score_1 = reweighting( (1-alpha_1), args.kappa)*weighted_style_score_1
        
        loss_1 = reweighted_content_score_1 + reweighted_style_score_1
        loss = loss_1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prog_bar.set_description("Loss:{:.5f} | Content loss:{:.5f} | Style loss:{:.5f}".format(
                                                        loss.item(),
                                                        (content_weight*content_score_1).item(),
                                                        (style_weight*style_score_1).item(),
                                                        ))

        """ generation result """
        dec_result_1 = input_img_1.clone().detach()
    

        if (total_step + 1) % 10000== 0:
            print("*Saving checkpoints")
            state_dict = network.state_dict()
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(state_dict, os.path.join(args.save_dir, str(total_step+1) + '_' + c_name+'_'+s_name+'_model.pth'))

            np.save(os.path.join(args.save_dir, str(total_step+1) + '_' + 'z_c'), z_c.clone().detach().cpu().numpy())
            np.save(os.path.join(args.save_dir, str(total_step+1) + '_' + 'z_s'), z_s.clone().detach().cpu().numpy())
          
            print('*CHECKPOINTS SAVED to ',os.path.join(args.save_dir, str(total_step+1) + '_' + c_name+'_'+s_name+ '_model.pth'))

        """ save the results """
        if (total_step+1) % 10000 == 0:
            result_1 = ( np.clip(( dec_result_1[0].permute(1,2,0).clone().detach().cpu().numpy()), 0.0, 1.0)*255.0).astype('uint8')[::,::,::-1]
          
            """ change the form """
            content_save  =  (cv2.resize(content_original, (args.img_size,args.img_size))).astype('uint8')[::,::,::-1]
            style_save    =  (cv2.resize(style_original, (args.img_size,args.img_size))).astype('uint8')[::,::,::-1]
            bundle_result =  np.stack( (content_save, style_save, result_1), axis=1 )
            bundle_result = bundle_result.reshape((args.img_size, args.img_size*3, 3))

            """ save the result """
            cv2.imwrite( os.path.join(args.save_dir, 'test', c_name+'_'+s_name+'_'+ str(total_step+1)+'result_1.jpg'), result_1)
            cv2.imwrite( os.path.join(args.save_dir, 'test', c_name+'_'+s_name+'_'+ str(total_step+1)+'result_bundle.jpg'), bundle_result)

    print(f"{args.type} time :", time.time() - start)
       
    return result  


if __name__ == "__main__":
    # parse options
    random_seed = 1006
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    """ start iteration """
    torch.cuda.empty_cache() # remove all caches

    #take one image in the list
        
    """ img load """
    print('*content_path :',content_path)
    print('*style_path :', style_path)
    content = img_load(content_path) # 1 image
    style  = img_load(style_path)
    content_original =content.copy()
    style_original = style.copy()
  
    """ Convert numpy array to PIL.Image format """
    """ and modify the range (0-255) to [0-1] ??""" 
    content = toPIL(content)
    style = toPIL(style)
   
    transform_list = []
    img_size = (args.img_size, args.img_size)
    transform_list.append(transforms.Resize(img_size, interpolation=2)) 
    transform_list += [transforms.ToTensor()]
    transform = transforms.Compose(transform_list)

    content = transform(content)
    style = transform(style)
    content = torch.unsqueeze( content, dim=0 )
    style   = torch.unsqueeze( style, dim=0 )

    if "cuda" in args.device:
        content = content.type(torch.cuda.FloatTensor).to(args.device)
        style = style.type(torch.cuda.FloatTensor).to(args.device)
    else:
        content = content.type(torch.FloatTensor).to(args.device)
        style = style.type(torch.FloatTensor).to(args.device)
    cnn = mods.vgg19(pretrained=True).features.to(args.device).eval()

    network, optimizer, coords = get_network(args,args.img_size)
    
    latent_size = args.latent_size
    z_c = torch.normal( 0, 1. , size=(latent_size,), device=args.device)
    z_s = torch.normal( 0, 1. , size=(latent_size,), device=args.device)

    if(args.start_iter > 0):
        network.load_state_dict(torch.load( os.path.join(args.save_dir, str(args.start_iter) + '_' + c_name+'_'+s_name+ '_model.pth') ))
        print('*Model Parameters Loaded')        
      
    output = run_style_transfer_inr(cnn,network,optimizer, cnn_normalization_mean,cnn_normalization_std, content,style,coords,content_original,style_original,z_c,z_s,num_steps=args.iter,style_weight=args.style_wt, content_weight=args.content_wt)

    del cnn
    gc.collect()
    