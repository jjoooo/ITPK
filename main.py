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
parser.add_argument("--n_patch",type=int,default=100)
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

root = '/Users/jui/Downloads/Data/'
data_name = 'MICCAI2008' #'MICCAI2008' #BRATS -> 'BRATS2015'
n_channel = 1
out_dim = 3
n_class = args.n_class
n_mode = args.n_mode # include gound truth
volume_size = (args.volume_size,args.volume_size,args.volume_size)
patch_size = (args.patch_size, args.patch_size, args.patch_size)
batch_size = args.batch_size
n_patch = args.n_patch

# Preprocessing
pp = Preprocessing(n_mode, n_class, n_patch/n_class, volume_size, patch_size, True, 'MICCAI2008', root, True)

p_path, l_path = pp.preprocess()

# Training (each mode)

unet = [[],[],[],[]]
test_path = pp.test_im_path()

for m in range(n_mode-1):
    # net
    unet[m] = nn.DataParallel(UnetGenerator_3d(in_dim=3,out_dim=out_dim,num_filter=16)).cuda()
    optimizer = torch.optim.Adam(unet[m].parameters(),lr=0.0002)

    cnt = 1
    output_cnt = 1
    
    x = Variable(torch.ones(batch_size,1,args.patch_size,args.patch_size,args.patch_size)).cuda()
    y = Variable(torch.zeros(batch_size,1,args.patch_size,args.patch_size,args.patch_size)).cuda()

    fn = ''
    if m==0: fn='FLAIR'
    elif m==1: fn='T1'
    else: fn='T2'

    while True:
        m_p_path = glob(p_path+'/{}_{}.mha'.format(cnt-1, m))
        m_l_path = glob(l_path+'/{}_l.mha'.format(cnt-1))
        if not m_p_path or not m_l_path: break

        p = io.imread(m_p_path, plugin='simpleitk').astype(float)
        l = io.imread(m_l_path, plugin='simpleitk').astype(float)

        for pp, ll in zip(p,l):
            x[cnt-1] = torch.from_numpy(pp)
            y[cnt-1] = torch.from_numpy(ll)
            if cnt % batch_size == 0:
                output = unet[m].forward(x)
                #label = Variable(torch.zeros(batch_size,1,16,16,16).type_as(torch.LongTensor())).cuda()
                loss = loss_function(output,y)
                
                loss.backward()
                optimizer.step()

                output_cnt += 1
                cnt = 1

            cnt += 1

            if output_cnt % 50 ==0:
                print('[MODE {}] - {}-------> loss : {}'.format(m, output_cnt, loss))
                torch.save(unet[m].state_dict(),'./model/miccai_{}_{}.pkl'.format(m, output_cnt))
        
    # test
    
    ct = 0
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