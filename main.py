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
parser.add_argument("--patch_size",type=int,default=32)
parser.add_argument("--n_patch",type=int,default=10000)
parser.add_argument("--batch_size",type=int,default=32)
parser.add_argument("--root",type=str,default='/mnt/disk1/data/MRI_Data/')
parser.add_argument("--data_name",type=str,default='MICCAI2008')
parser.add_argument("--n_class",type=int,default=2)
parser.add_argument("--n_mode",type=int,default=4)
parser.add_argument("--volume_size",type=int,default=512)
parser.add_argument("--learning_rate",type=float,default=0.001)
parser.add_argument("--fold_val",type=int,default=5)
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
lr = args.learning_rate
fold_val = args.fold_val

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
print('learning rate = {}'.format(lr))
print('N fold validation = {}'.format(fold_val))
print('----------------------------------------------')

# Preprocessing
tr = Preprocessing(n_mode, n_class, n_patch/n_class, volume_size, patch_size, False, False, data_name, root, True)
# test = Preprocessing(n_mode, n_class, n_patch/n_class, volume_size, patch_size, False, False, data_name, root, False)

print('Create training patches')
p_path, l_path = tr.preprocess()
print('Done.')
# print('Create test patches')
# test_p_path = test.test_preprocess()
# print('Done.')

# Training
# net
unet = nn.DataParallel(UnetGenerator_3d(in_dim=n_mode-1,out_dim=out_dim,num_filter=16)).cuda()
optimizer = torch.optim.Adam(unet.parameters(),lr=lr)

file = open('./lr{}_ps{}_mse_loss'.format(lr,patch_size[0]), 'w')
file_dsc = open('./lr{}_ps{}_DSC'.format(lr,patch_size[0]), 'w')

model_path = './model/model_ps{}_bs{}_np{}_lr{}'.format(args.patch_size, batch_size, n_patch, lr)
if not os.path.exists(model_path):
    os.makedirs(model_path)

all_num_patches = len(glob(p_path+'/**'))
tr_num = int(all_num_patches*(float(fold_val-1)/float(fold_val)))
val_num = all_num_patches - tr_num
print('{} fold-validation : tr={}, test={}, in {}'.format(fold_val, tr_num, all_num_patches-tr_num, all_num_patches))

DSC = 0.0
for it in range(fold_val):
    cnt = 1
    output_cnt = 1

    val_idx_start = it * val_num
    val_idx_end = val_idx_start + val_num
    print('validation patch idx : {} - {}'.format(val_idx_start, val_idx_end))
    patch_n = 0
    
    while True:
        if patch_n >= all_num_patches: break
        if len(glob(model_path+'/**'))>0: break
        if patch_n < val_idx_start or patch_n >= val_idx_end:
            
            m_p_path = glob(p_path+'/{}.mha'.format(patch_n))
            m_l_path = glob(l_path+'/{}_l.mha'.format(patch_n))
            if not m_p_path or not m_l_path: break

            x = np.zeros([batch_size, n_mode-1, args.patch_size, args.patch_size, args.patch_size])
            y = np.zeros([batch_size, n_channel, args.patch_size, args.patch_size, args.patch_size])

            p = io.imread(m_p_path[0], plugin='simpleitk').astype(float)
            l = io.imread(m_l_path[0], plugin='simpleitk').astype(float) 

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
                file.write(str(loss)+"\n")
                print('batch {}\nloss {}'.format(output_cnt-1, loss))

                loss.backward()
                optimizer.step()

                output_cnt += 1
                cnt = 1

            cnt += 1
            
            if output_cnt % 1000 ==0:
                torch.save(unet.state_dict(),model_path+'/miccai_{}.pkl'.format(output_cnt))

        patch_n += 1

    cnt = 1
    output_cnt = 1
    patch_n = val_idx_start
    dice = 0.0

    while True:
        if patch_n >= all_num_patches: break
        if patch_n >= val_idx_start and patch_n < val_idx_end:
            
            m_p_path = glob(p_path+'/{}.mha'.format(patch_n))
            m_l_path = glob(l_path+'/{}_l.mha'.format(patch_n))
            if not m_p_path or not m_l_path: break

            x = np.zeros([batch_size, n_mode-1, args.patch_size, args.patch_size, args.patch_size])
            y = np.zeros([batch_size, n_channel, args.patch_size, args.patch_size, args.patch_size])

            p = io.imread(m_p_path[0], plugin='simpleitk').astype(float)
            l = io.imread(m_l_path[0], plugin='simpleitk').astype(float) 

            for m in range(n_mode-1):
                d1 = m*args.patch_size
                d2 = (m+1)*args.patch_size
                x[cnt-1,m] = p[d1:d2]
            y[cnt-1,0] = l

            if cnt % batch_size == 0:
                x = Variable(torch.from_numpy(x).float()).cuda()
                y = Variable(torch.from_numpy(y).long()).cuda()
                output = unet.forward(x)
                
                # DSC
                y_arr = y.cpu().numpy()
                output_arr = output.cpu().numpy()

                TP = np.sum(output_arr[y_arr==1])
                dice += TP*2.0 / (np.sum(output_arr) + np.sum(y_arr))

                output_cnt += 1
                cnt = 1
            cnt += 1
        patch_n += 1
    
    DSC += dice/(output_cnt-1)
    file_dsc.write('{} fold-validation : DSC={}\n'.format(it, DSC))

DSC = DSC/fold_val
file_dsc.write("=================>>>> Result={}\n".format(DSC))

'''
# test

# model loading
im_path = glob(test_p_path + '/**')
models_path = glob('./model/model_ps{}_bs{}_np{}_lr{}/miccai_2300.pkl'.format(args.patch_size, batch_size, n_patch, lr))

if not os.path.isfile(models_path):
    print(models_path+' -> model not exists')

for idx, im in enumerate(im_path):

    unet.load_state_dict(torch.load(models_path))
    output_volume = np.zeros([volume_size[0], volume_size[1], volume_size[2]])

    for dd in range(volume_size[0]):
        for hh in range(volume_size[1]):
            for ww in range(volume_size[2]):
                d1 = z-int(patch_size[0]/2)
                d2 = z+int(patch_size[0]/2)
                h1 = y-int(patch_size[1]/2)
                h2 = y+int(patch_size[1]/2)
                w1 = x-int(patch_size[2]/2)
                w2 = x+int(patch_size[2]/2)

                if d1 < 0 or d2 > d or h1 < 0 or h2 > h or w1 < 0 or w2 > w:
                    continue
                    
                p = test_p_path+'/{}_{}_{}.mha'.format(dd,hh,ww)
                if not os.path.isfile(p):
                    print(p+' -> not exists')
                    continue

                x = np.zeros([1, n_mode-1, args.patch_size, args.patch_size, args.patch_size])
                patch = io.imread(p, plugin='simpleitk').astype(float)
                for m in range(n_mode-1):
                    d1 = m*args.patch_size
                    d2 = (m+1)*args.patch_size
                    x[cnt-1,m] = p[d1:d2]
                x = Variable(torch.from_numpy(x).float()).cuda()
                output = unet.forward(x)
                for m in range(n_mode):
                    if m==0:
                        patch_mode = output
                    else:
                        patch_mode += output
                
                output_volume[d1:d2, h1:h2, w1:w2] = patch_mode

    # save
    thsd = 3 # max = n_mode+9 
    output_volume = (output_volume>thsd)*1
    path = './test/{}'.format(idx)
    if not os.path.exists(path):
        os.makedirs(path)
    s_cnt = 0
    for slice in output_volume:
        io.imsave(path+'/{}.PNG'.format(s_cnt), slice)    
        s_cnt += 1
   '''
