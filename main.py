import os
import numpy as np
from glob import glob
import random

# image data
from skimage import io
import SimpleITK as sitk

from preprocessing import Preprocessing

# Learning
import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from model import UnetGenerator_3d, loss_function

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_gpu",type=int,default=8)
parser.add_argument("--patch_size",type=int,default=16)
parser.add_argument("--n_patch",type=int,default=10)
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--root",type=str,default='/Users/jui/Downloads/Data/')
parser.add_argument("--data_name",type=str,default='MICCAI2008')
parser.add_argument("--n_class",type=int,default=2)
parser.add_argument("--n_mode",type=int,default=4)
parser.add_argument("--volume_size",type=int,default=512)
args = parser.parse_args()

import warnings
warnings.filterwarnings("ignore")
    
try:
    xrange
except NameError:
    xrange = range

root = args.root
data_name = args.data_name #'MICCAI2008' #BRATS -> 'BRATS2015'
n_channel = 1
out_dim = 2
n_class = args.n_class
n_mode = args.n_mode # include gound truth
if data_name == 'BRATS2015':
    volume_size = (155,args.volume_size,args.volume_size)
else:
    volume_size = (args.volume_size,args.volume_size,args.volume_size)
patch_size = (args.patch_size, args.patch_size, args.patch_size)
batch_size = args.batch_size
n_patch = args.n_patch

print('----------------------------------------------')
print('n_gpu = {}'.format(args.n_gpu))
print('volume size = {}'.format(volume_size))
print('patch size = {}'.format(patch_size))
print('batch size = {}'.format(batch_size))
print('n_channel = {}'.format(n_channel))
print('n_class = {}'.format(n_class))
print('n_mode = {}'.format(n_mode))
print('n_patches = {}'.format(n_patch))
print('root = '+root)
print('data name = '+data_name)
print('----------------------------------------------')

# Preprocessing
pp = Preprocessing(n_mode, n_class, n_patch/n_class, volume_size, patch_size, False, False, data_name, root, True)

p_path, l_path = pp.preprocess()

# Training
# net
unet = nn.DataParallel(UnetGenerator_3d(in_dim=n_mode,out_dim=out_dim,num_filter=16)).cuda()
optimizer = torch.optim.Adam(unet.parameters(),lr=0.0002)

cnt = 1
output_cnt = 1
patch_n = 0

patch_path = glob(p_path+'/**')
label_path = glob(l_path+'/**')

for pp,lp in zip(patch_path, label_path):

    x = np.zeros([batch_size, n_mode-1, args.patch_size, args.patch_size, args.patch_size])
    y = np.zeros([batch_size, n_channel, args.patch_size, args.patch_size, args.patch_size])

    p = io.imread(pp, plugin='simpleitk').astype(float)
    l = io.imread(lp, plugin='simpleitk').astype(float)
    
    for m in range(n_mode-1):
        d1 = m*args.patch_size
        d2 = (m+1)*args.patch_size
        x[cnt-1,m] = p[d1:d2]
    y[cnt-1,0] = l

    if cnt % batch_size == 0:
        x = Variable(torch.from_numpy(x).float()).cuda()
        y = Variable(torch.from_numpy(y).long()).cuda()
        output = unet.forward(x)
        loss = loss_function(output,y)
        
        loss.backward()
        optimizer.step()

        output_cnt += 1
        cnt = 1

    cnt += 1

    model_path = './model'
    if not os.path.exists(model_path):
            os.makedirs(model_path)
    if output_cnt % 50 ==0:
        print('[{}] -------> loss : {}'.format(output_cnt, loss))
        torch.save(unet.state_dict(),'./model/miccai_{}.pkl'.format(output_cnt))

    patch_n += 1
    
'''
# test
test_path = pp.test_im_path()

ct = 0
for p in test_path:
    x = io.imread(p, plugin='simpleitk').astype(float)
    y = unet.forward(x)

    # save
    path = './test/'
    if not os.path.exists(path):
        os.makedirs(path)
    s_cnt = 0
    for slice in y:
        io.imsave(path+'{}_{}.PNG'.format(cnt, s_cnt), slice)    
        s_cnt += 1
    ct += 1


# model loading

for m in range(n_mode-1):
    models_path = glob('./model/miccai_{}_*.pkl'.format(m))
    model_idx = len(models_path)
    unet[m].load_state_dict(torch.load('./model/miccai_{}_{}.pkl'.format(m, model_idx)))
    for p in test_path[m]:
        x = io.imread(p, plugin='simpleitk').astype(float)
        y = unet[m].forward(x)

        # save
        path = './test/'
        if not os.path.exists(path):
            os.makedirs(path)
        s_cnt = 0
        for slice in y:
            io.imsave(path+'{}_{}_{}.PNG'.format(m, cnt, s_cnt), slice)    
            s_cnt += 1
        ct += 1
'''