import random
import os
import argparse
import cv2
import random
import time
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

from models.mlp import *
from models.utils import *

parser = argparse.ArgumentParser(description='Code to optimize')
parser.add_argument('--device', help='cuda | cuda:0 | cpu', default="cuda", type=str)
parser.add_argument('--device_num', help='which GPUs to use', default="0",  type=str)

parser.add_argument('--content_path', required=True, type=str, help='path to content image')
parser.add_argument('--style_path', required=True, type=str, help='path to style image')
parser.add_argument('--save_dir', default='./checkpoints',help='Directory to save the model')
parser.add_argument('--start_iter', type=int, default=20000)

parser.add_argument('--latent_size', default=16, type=int, help='latent vector size')
parser.add_argument('--img_size', help="size of input image", default=256, type=int)

parser.add_argument('--H', type=int, help='height')
parser.add_argument('--W', type=int, help='width')
parser.add_argument('--L', type=float, help='left')
parser.add_argument('--R', type=float, help='right')
parser.add_argument('--T', type=float, help='top')
parser.add_argument('--B', type=float, help='bottom')
parser.add_argument('--alpha', type=float, help='interpolation rate')

parser.add_argument('--depth', default=8, type=int, help='model depth')
parser.add_argument('--width', default=256, type=int, help='feature size')

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
    img = cv2.imread(path)[::,::,::-1] 
    return img

def toPIL(img):
    img_type = str(type(img))
    if 'numpy' in img_type:
        img = Image.fromarray(img)
    elif 'torch' in img_type:
        img = transforms.ToPILImage()(img).convert("RGB")
    return img


def get_mgrid_test(size_h, size_w, L,R,T,B, dim=2):
    size_h = torch.linspace(T,B, steps=size_h)
    size_w = torch.linspace(L,R, steps=size_w)
    
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

    os.makedirs(str(os.path.join(args.save_dir, 'size_control')), exist_ok=True)

    c_img = os.path.basename(content_path) 
    s_img = os.path.basename(style_path)   
    c_name = c_img.split('.')[0] 
    s_name = s_img.split('.')[0] 


    random_seed = 1006
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

   
    torch.cuda.empty_cache() 

    """ img load """
    print('*content_path :',content_path)
    print('*style_path :', style_path)
    content = img_load(content_path) 
    style  = img_load(style_path)
    content_original =content.copy()
    style_original = style.copy()
   
    content = toPIL(content)
    style = toPIL(style)
   
    transform_list = []
    img_size = (args.H, args.W)
    transform_list.append(transforms.Resize(img_size, interpolation=2)) 
    transform_list += [transforms.ToTensor()]
    transform = transforms.Compose(transform_list)

    content = transform(content)
    style = transform(style)
    content = torch.unsqueeze( content, dim=0 )
    style   = torch.unsqueeze( style, dim=0 )

    content = content.type(torch.FloatTensor).to(args.device)
    style = style.type(torch.FloatTensor).to(args.device)

    network, coords = get_network_eval(args, args.img_size)
    coords = get_mgrid_test(args.H, args.W, args.L,args.R,args.T,args.B).to(args.device)

    network.load_state_dict(torch.load(os.path.join(args.save_dir, str(args.start_iter) + '_' + c_name+'_'+s_name+'_model.pth')) )
    network.eval()
    print("num_params: ", sum(p.numel() for p in network.parameters()) )
    print('*Model Parameters Loaded')

    z_c = np.load(os.path.join(args.save_dir, str(args.start_iter) + '_z_c.npy') )
    z_s = np.load(os.path.join(args.save_dir, str(args.start_iter) + '_z_s.npy') )
    
    z_c = torch.from_numpy(z_c).to(args.device)
    z_s = torch.from_numpy(z_s).to(args.device)

    alpha = args.alpha

    batch_step = 1000*1200 
    if (coords.shape[0]%batch_step)>0:
        batch_size = (coords.shape[0]//batch_step) + 1
    else:
        batch_size = (coords.shape[0]//batch_step)

    z = (alpha * z_c + (1-alpha) * z_s).to(args.device)
    z = z.unsqueeze(0)
    
    print('z size:', z.size())
    print('coords size:', coords.size())
    torch.cuda.synchronize()
    start_time = time.time()
    print("start")
    output_list = []
    print("batch_size: ", batch_size)

    z_repeat = z.repeat(batch_step, 1)
    with torch.no_grad():
        for i in range(batch_size):

            try:
                new_coords = coords[i*batch_step:(i+1)*batch_step]
                output = network(new_coords, z_repeat) 
            except:
             
                new_coords = coords[i*batch_step:]
                z_repeat = z.repeat(new_coords.shape[0], 1)
                output = network(new_coords, z_repeat) 
   
            output_list.append(output.detach().cpu())

    output = torch.cat(output_list, dim=0)
    output = output.permute(1,0)
    output = output.view(-1,args.H,args.W)
    output = torch.unsqueeze( output, dim=0 )

    torch.cuda.synchronize()
    print("Max GPU memory allocated: %.4f GB" % (torch.cuda.max_memory_allocated(device=0) / 1024. / 1024. / 1024.))
    print("Total memory of the current GPU: %.4f GB" % (torch.cuda.get_device_properties(device=0).total_memory / 1024. / 1024 / 1024))
    print("inference time: ", time.time() - start_time)
         
    result = output
    result = ( np.clip(( result[0].permute(1,2,0).numpy()), 0.0, 1.0)*255.0).astype('uint8')[::,::,::-1]

    """ change the form """
    content_save  =  (cv2.resize(content_original, (args.H,args.W))).astype('uint8')[::,::,::-1]
    style_save    =  (cv2.resize(style_original, (args.H,args.W))).astype('uint8')[::,::,::-1]

    """ save the result """
    cv2.imwrite( os.path.join(args.save_dir, 'size_control',str(args.start_iter) +'_'+'size: '+f'{args.H}x{args.W}'+'_'+'alpha: '+ str(alpha) +'_'+'result.jpg'), result)
