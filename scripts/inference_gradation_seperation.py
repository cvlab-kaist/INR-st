import random
import os
import numpy as np
import torch
from PIL import Image
import argparse 

import os
import cv2
import random
import numpy as np

import torch
import torchvision.transforms as transforms

from models.mlp import *
from models.utils import *

parser = argparse.ArgumentParser(description='Code to optimize')
parser.add_argument('--device',                  help='cuda | cuda:0 | cpu',                   default="cuda", type=str)
parser.add_argument('--device_num',              help='which GPUs to use',                     default="3",  type=str)
parser.add_argument('--img_size',     help="size of input image",                  default=256, type=int)

parser.add_argument('--content_path', required=True, type=str, help='path to content image')
parser.add_argument('--style_path', required=True, type=str, help='path to style image')
parser.add_argument('--save_dir', default='./checkpoints',help='Directory to save the model')
parser.add_argument('--start_iter', type=int, default=20000)

parser.add_argument('--mode', required=True, type=str, help='mode')
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
    # tensors = tuple(size_h, size_w)
    mgrid = torch.stack(torch.meshgrid(size_h,size_w), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

def get_transform(img_size,transform_list):
    img_size = (img_size, img_size)
    transform_list.append(transforms.Resize(img_size)) # @@@@ args.interpol-method = 2
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

    os.makedirs(str(os.path.join(args.save_dir, 'latent_gradation')), exist_ok=True)

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
    transform_list.append(transforms.Resize(img_size, interpolation=2)) # @@@@ args.interpol-method = 2
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

    network, coords = get_network_eval(args, 256)
    network.eval()
    
    network.load_state_dict(torch.load(os.path.join(args.save_dir,str(args.start_iter) + '_' + c_name+'_'+s_name+'_model.pth')) )
    print('*Model Parameters Loaded')

    z_c = np.load(os.path.join(args.save_dir, str(args.start_iter) + '_z_c.npy') )
    z_s = np.load(os.path.join(args.save_dir, str(args.start_iter) + '_z_s.npy') )
    

    z_c = torch.from_numpy(z_c).to(args.device)
    z_s = torch.from_numpy(z_s).to(args.device)

    #coord = convert_to_coord_format(z_c.size(0), 256, 256, args.device, integer_values=False)
    
    alphas = [0.0]
    mode = args.mode

    if mode == 'gradation' :
        z = z_s.view(1,16)        
        print(z.size())
        z = z.repeat(args.img_size*args.img_size, 1)
        print(z.size())
        z = z.view(args.img_size,args.img_size,16)
        
        g = torch.linspace(1,0, steps=args.img_size)
        for i in range(z.size(1)):
            if i < z.size(1)/2:
                z[:,i,:] = (g[i] * z_c + (1-g[i]) * z_s).to(args.device)
            elif i >= z.size(1)/2:
                z[:,i,:] = (g[i] * z_c + (1-g[i]) * z_s).to(args.device)

        z = z.view(args.img_size*args.img_size,16)

        output = network(coords, z) # DeepSDF
        output = output.permute(1,0)
        output = output.view(-1,args.img_size,args.img_size)
        output = torch.unsqueeze( output, dim=0 )
        output = output.type(torch.cuda.FloatTensor).to(args.device)        
        result = output.clone().detach()
        result = ( np.clip(( result[0].permute(1,2,0).clone().detach().cpu().numpy()), 0.0, 1.0)*255.0).astype('uint8')[::,::,::-1]
        
        """ save the result """
        cv2.imwrite( os.path.join(args.save_dir, 'latent_gradation', c_name+'_'+s_name+'_' + 'LtoR.jpg'), result)

        z = z_s.view(1,16)        
        print(z.size())
        z = z.repeat(args.img_size*args.img_size, 1)
        print(z.size())
        z = z.view(args.img_size,args.img_size,16)
        
        g = torch.linspace(0,1, steps=args.img_size)
        for i in range(z.size(1)):
            z[i,:,:] = (g[i] * z_c + (1-g[i]) * z_s).to(args.device)
        
        z = z.view(args.img_size*args.img_size,16)

        output = network(coords, z) # DeepSDF
        output = output.permute(1,0)
        output = output.view(-1,args.img_size,args.img_size)
        output = torch.unsqueeze( output, dim=0 )
        output = output.type(torch.cuda.FloatTensor).to(args.device)        
        result = output.clone().detach()
        result = ( np.clip(( result[0].permute(1,2,0).clone().detach().cpu().numpy()), 0.0, 1.0)*255.0).astype('uint8')[::,::,::-1]
        
        """ save the result """
        cv2.imwrite( os.path.join(args.save_dir, 'latent_gradation', c_name+'_'+s_name+'_' + 'TtoB.jpg'), result)
        
    elif mode == 'seperate_col' :
        z = z_s.view(1,16)        
        print(z.size())
        z = z.repeat(args.img_size*args.img_size, 1)
        print(z.size())
        z = z.view(args.img_size,args.img_size,16)
        
        g = torch.linspace(0,1, steps=args.img_size)
        for i in range(z.size(1)):
            if i/args.img_size <= 0.25: z[:,i,:] = (0.25 * z_c + (1-0.25) * z_s).to(args.device)
            elif 0.25 < i/args.img_size and i/args.img_size <=0.5: z[:,i,:] = (1.0 * z_c + (1-1.0) * z_s).to(args.device)
            elif 0.5 < i/args.img_size and i/args.img_size <= 0.75: z[:,i,:] = (0 * z_c + (1-0) * z_s).to(args.device)
            elif i/args.img_size > 0.75: z[:,i,:] = (0.75 * z_c + (1-0.75) * z_s).to(args.device)
            
        
        z = z.view(args.img_size*args.img_size,16)

        output = network(coords, z) # DeepSDF
        output = output.permute(1,0)
        output = output.view(-1,args.img_size,args.img_size)
        output = torch.unsqueeze( output, dim=0 )
        output = output.type(torch.cuda.FloatTensor).to(args.device)        
        result = output.clone().detach()
        result = ( np.clip(( result[0].permute(1,2,0).clone().detach().cpu().numpy()), 0.0, 1.0)*255.0).astype('uint8')[::,::,::-1]
        
        """ save the result """
        cv2.imwrite( os.path.join(args.save_dir, 'latent_gradation', c_name+'_'+s_name+'_' + 'separate_col.jpg'), result)

    elif mode == 'seperate_row' :
        z = z_s.view(1,16)        
        print(z.size())
        z = z.repeat(args.img_size*args.img_size, 1)
        print(z.size())
        z = z.view(args.img_size,args.img_size,16)
        
        g = torch.linspace(0,1, steps=args.img_size)
        for i in range(z.size(0)):
            if i/args.img_size <= 0.25: z[i,:,:] = (0.25 * z_c + (1-0.25) * z_s).to(args.device)
            elif 0.25 < i/args.img_size and i/args.img_size <=0.5: z[i,:,:] = (1.0 * z_c + (1-1.0) * z_s).to(args.device)
            elif 0.5 < i/args.img_size and i/args.img_size <= 0.75: z[i,:,:] = (0 * z_c + (1-0) * z_s).to(args.device)
            elif i/args.img_size > 0.75: z[i,:,:] = (0.75 * z_c + (1-0.75) * z_s).to(args.device)
            
        
        z = z.view(args.img_size*args.img_size,16)

        output = network(coords, z) # DeepSDF
        output = output.permute(1,0)
        output = output.view(-1,args.img_size,args.img_size)
        output = torch.unsqueeze( output, dim=0 )
        output = output.type(torch.cuda.FloatTensor).to(args.device)        
        result = output.clone().detach()
        result = ( np.clip(( result[0].permute(1,2,0).clone().detach().cpu().numpy()), 0.0, 1.0)*255.0).astype('uint8')[::,::,::-1]
        
        """ save the result """
        cv2.imwrite( os.path.join(args.save_dir, 'latent_gradation', c_name+'_'+s_name+'_' + 'separate_row.jpg'), result)


    elif mode == 'checkboard' :
        z = z_s.view(1,16)        
        print(z.size())
        z = z.repeat(65536, 1)
        print(z.size())
        z = z.view(256,256,16)
        
        g = torch.linspace(0,1, steps=args.img_size)
        for i in range(z.size(1)):
            if i/args.img_size <= 0.25: z[:,i,:] = (0.25 * z_c + (1-0.25) * z_s).to(args.device)
            elif 0.25 < i/args.img_size and i/args.img_size <=0.5: z[:,i,:] = (1.0 * z_c + (1-1.0) * z_s).to(args.device)
            elif 0.5 < i/args.img_size and i/args.img_size <= 0.75: z[:,i,:] = (0 * z_c + (1-0) * z_s).to(args.device)
            elif i/args.img_size > 0.75: z[:,i,:] = (0.75 * z_c + (1-0.75) * z_s).to(args.device)
        
        for i in range(z.size(1)):
            if i/args.img_size <= 0.25: z[:,i,:] = (0.25 * z_c + (1-0.25) * z_s).to(args.device)
            elif 0.25 < i/args.img_size and i/args.img_size <=0.5: z[:,i,:] = (1.0 * z_c + (1-1.0) * z_s).to(args.device)
            elif 0.5 < i/args.img_size and i/args.img_size <= 0.75: z[:,i,:] = (0.5 * z_c + (1-0.5) * z_s).to(args.device)
            elif i/args.img_size > 0.75: z[:,i,:] = (0.75 * z_c + (1-0.75) * z_s).to(args.device)
        
        z = z.view(256*256,16)

        output = network(coords, z) # DeepSDF
        output = output.permute(1,0)
        output = output.view(-1,args.img_size,args.img_size)
        output = torch.unsqueeze( output, dim=0 )
        output = output.type(torch.cuda.FloatTensor).to(args.device)        
        result = output.clone().detach()
        result = ( np.clip(( result[0].permute(1,2,0).clone().detach().cpu().numpy()), 0.0, 1.0)*255.0).astype('uint8')[::,::,::-1]
        
        """ save the result """
        cv2.imwrite( os.path.join(args.save_dir, 'latent_gradation', c_name+'_'+s_name+'_' + 'separate.jpg'), result)