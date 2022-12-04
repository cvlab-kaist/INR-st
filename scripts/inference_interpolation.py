import random
import os
import cv2
import numpy as np
from PIL import Image
import argparse

import torch
import torchvision.transforms as transforms

from models.mlp import *
from models.utils import *

parser = argparse.ArgumentParser(description='Code to optimize')
parser.add_argument('--device',                  help='cuda | cuda:0 | cpu',                   default="cuda", type=str)
parser.add_argument('--device_num',              help='which GPUs to use',                     default="0",  type=str)

parser.add_argument('--img_size',     help="size of input image",                  default=256, type=int)
parser.add_argument('--content_path', required=True, type=str, help='path to content image')
parser.add_argument('--style_path', required=True, type=str, help='path to style image')
parser.add_argument('--save_dir', default='./checkpoints',help='Directory to save the model')
parser.add_argument('--start_iter', type=int, default=20000)

parser.add_argument('--latent_size',default=16, type=int, help='size of latent vector')
parser.add_argument('--depth',default=8, type=int, help='model depth')
parser.add_argument('--width',default=256, type=int, help='feature size')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
GPU_list = args.device_num
len_GPU = len( GPU_list.split(","))
print("*len_GPU: ", len_GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_list


mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
std =  np.array([0.229, 0.224, 0.225]).reshape(1,1,3)

#  Load content and style
def img_load(path):
    img = cv2.imread(path)[::,::,::-1] # BGR to RGB, [0-255]
    return img

def toPIL(img):
    # image range should be [0-255] for converting.
    img_type = str(type(img))
    if 'numpy' in img_type:
        img = Image.fromarray(img)
    elif 'torch' in img_type:
        img = transforms.ToPILImage()(img).convert("RGB")
    return img


def get_mgrid_test(size_h, size_w, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    size_h = torch.linspace(-2, 2, steps=size_h)
    size_w = torch.linspace(-2, 2, steps=size_w)

    mgrid = torch.stack(torch.meshgrid(size_h,size_w), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def get_transform(img_size,transform_list):
    img_size = (img_size, img_size)
    transform_list.append(transforms.Resize(img_size)) 
    transform_list += [transforms.ToTensor()]
    transform = transforms.Compose(transform_list) 
    return transform

def normalization(img_tensor):
    vggMean = torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1).type(torch.FloatTensor)
    vggStd  =  torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1).type(torch.FloatTensor)
    img_tensor = (img_tensor-vggMean)/vggStd
    return img_tensor

if __name__ == "__main__":
  
    content_path =   args.content_path
    style_path   =   args.style_path

    os.makedirs(str(os.path.join(args.save_dir, 'latent_interpolation')), exist_ok=True)

    c_img = os.path.basename(content_path) 
    s_img = os.path.basename(style_path) 
    c_name = c_img.split('.')[0]
    s_name = s_img.split('.')[0]

    if "cuda" in args.device:
        vggMean = torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1).type(torch.cuda.FloatTensor).to(args.device)
        vggStd  =  torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1).type(torch.cuda.FloatTensor).to(args.device)
    else:
        vggMean = torch.tensor([0.485, 0.456, 0.406]).view(-1,1,1).type(torch.FloatTensor).to(args.device)
        vggStd  =  torch.tensor([0.229, 0.224, 0.225]).view(-1,1,1).type(torch.FloatTensor).to(args.device)
        
    random_seed = 1006
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

    """ start iteration """
    torch.cuda.empty_cache() # remove all caches

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


    network,coords = get_network_eval(args, args.img_size)
    network.eval()
    network.load_state_dict(torch.load(os.path.join(args.save_dir, str(args.start_iter) + '_' + c_name+'_'+s_name+'_model.pth')) )
    print('*Model Parameters Loaded')

    z_c = np.load(os.path.join(args.save_dir, str(args.start_iter) + '_z_c.npy') )
    z_s = np.load(os.path.join(args.save_dir, str(args.start_iter) + '_z_s.npy') )
    
    z_c = torch.from_numpy(z_c).to(args.device)
    z_s = torch.from_numpy(z_s).to(args.device)

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for alpha in alphas:
        z = (alpha * z_c + (1-alpha) * z_s).to(args.device)

        output = network(coords, z.repeat(args.img_size*args.img_size, 1)) # DeepSDF
        output = output.permute(1,0)
        output = output.view(-1,args.img_size,args.img_size)
        output = torch.unsqueeze( output, dim=0 )

        output = output.type(torch.cuda.FloatTensor).to(args.device)        
        result = output.clone().detach()
        result = ( np.clip(( result[0].permute(1,2,0).clone().detach().cpu().numpy()), 0.0, 1.0)*255.0).astype('uint8')[::,::,::-1]
        
        """ change the form """
        content_save  =  (cv2.resize(content_original, (args.img_size,args.img_size))).astype('uint8')[::,::,::-1]
        style_save    =  (cv2.resize(style_original, (args.img_size,args.img_size))).astype('uint8')[::,::,::-1]
        bundle_result =  np.stack( (content_save, style_save, result), axis=1 )
        bundle_result = bundle_result.reshape((args.img_size, args.img_size*3, 3))

        """ save the result """
        cv2.imwrite( os.path.join(args.save_dir, 'latent_interpolation', c_name+'_'+s_name+'_'+ str(args.start_iter) +'_'+'alpha: '+str(alpha)+'_'+'result.jpg'), result)
    