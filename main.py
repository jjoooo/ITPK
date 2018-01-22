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
import torch.utils as utilss
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from model import UnetGenerator_3d, loss_function
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_idx",type=int,default=0)
parser.add_argument("--n_epoch",type=int,default=100)
parser.add_argument("--patch_size",type=int,default=32)
parser.add_argument("--n_patch",type=int,default=10000)
parser.add_argument("--batch_size",type=int,default=1024)
parser.add_argument("--root",type=str,default='/mnt/disk2/data/MRI_Data/')
parser.add_argument("--data_name",type=str,default='MICCAI2008')
parser.add_argument("--n_class",type=int,default=2)
parser.add_argument("--n_mode",type=int,default=4)
parser.add_argument("--volume_size",type=int,default=512)
parser.add_argument("--learning_rate",type=float,default=0.0002)
parser.add_argument("--fold_val",type=int,default=5)
parser.add_argument("--train_bl",type=int,default=1)
args = parser.parse_args()

import warnings
warnings.filterwarnings("ignore")
    
try:
    xrange
except NameError:
    xrange = range

n_epoch = args.n_epoch
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
train_bool = True
if args.train_bl == 0:
    train_bool = False
n4b = False
use_gpu = '{},{}'.format(args.gpu_idx, args.gpu_idx+1)
os.environ["CUDA_VISIBLE_DEVICES"]=use_gpu


print('----------------------------------------------')
print('use_gpu = '+use_gpu)
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
print('Training = {}'.format(train_bool))
print('----------------------------------------------')

if not os.path.exists('./loss'):
    os.makedirs('./loss')
if not os.path.exists('./dsc'):
    os.makedirs('./dsc')

file_loss = open('./loss/lr{}_bs{}_ps{}_np{}_{}fold_mse_loss'.format(lr,batch_size,patch_size[0],n_patch,fold_val), 'w')
file_dsc = open('./dsc/lr{}_bs{}_ps{}_np{}_{}fold_DSC'.format(lr,batch_size,patch_size[0],n_patch,fold_val), 'w')
test_file_dsc = open('./dsc/TEST_lr{}_bs{}_ps{}_np{}_{}fold_DSC'.format(lr,batch_size,patch_size[0],n_patch,fold_val), 'w')

model_path = './model/model_ps{}_np{}_lr{}_{}fold'.format(args.patch_size, n_patch, lr, fold_val)

if train_bool:
    # Preprocessing
    tr = Preprocessing(n_mode, n_class, n_patch, volume_size, patch_size, n4b, True, data_name, root, train_bool)

    print('\nCreate patch for training...')
    p_path, l_path = tr.preprocess()
    print('Done.\n')

    # Training
    # net
    unet = nn.DataParallel(UnetGenerator_3d(in_dim=n_mode-1,out_dim=out_dim,num_filter=16)).cuda()
    optimizer = torch.optim.Adam(unet.parameters(),lr=lr)
    optimizer.zero_grad()
    
    output_cnt = 1
    # exist models
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    models_path = glob(model_path+'/**')
    if models_path:
        md_path = model_path+'/miccai_{}.pkl'.format(len(models_path)*100)
        output_cnt = len(models_path)*100+1

        if not os.path.isfile(md_path):
            print(md_path+' -> model not exists\n')
            md_path = model_path+'/miccai_{}.pkl'.format(len(models_path)*1000)
            output_cnt = len(models_path)*1000+1
            if not os.path.isfile(md_path):
                print(md_path+' -> also this model not exists\n')
                output_cnt = 1
            else:
                print('pretrained model loading: '+md_path)
                unet.load_state_dict(torch.load(md_path))
        else:
            print('pretrained model loading: '+md_path)
            unet.load_state_dict(torch.load(md_path))

    all_num_patches = len(glob(p_path+'/**'))
    tr_num = int(all_num_patches*(float(fold_val-1)/float(fold_val)))
    val_num = all_num_patches - tr_num
    print('{} fold-validation : tr={}, test={}, in {}'.format(fold_val, tr_num, all_num_patches-tr_num, all_num_patches))

    DSC = 0.0
    minDice = 1.0
    maxDice = 0.0

    for ep in range(n_epoch):
        for it in range(fold_val):
            cnt = 1

            val_idx_start = it * val_num
            val_idx_end = val_idx_start + val_num
            print('{} fold validation patch idx : {} - {}'.format(it, val_idx_start, val_idx_end))
            patch_n = 0
            
            while True:
                if patch_n >= all_num_patches:  
                    print('Training done.\n')
                    break
                # if len(glob(model_path+'/**'))>0: break
                if patch_n < val_idx_start or patch_n >= val_idx_end:
                #if patch_n >= val_idx_start and patch_n < val_idx_end:
                    m_p_path = glob(p_path+'/{}.mha'.format(patch_n))
                    m_l_path = glob(l_path+'/{}_l.mha'.format(patch_n))
                    if not m_p_path or not m_l_path: 
                        print('ERR file not exists')
                        patch_n += 1
                        continue

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
                        file_loss.write('batch {} \t: loss {}\n'.format(output_cnt-1, loss.data.cpu().numpy()[0]))
                        print('batch {} \t-------> loss {}'.format(output_cnt-1, loss.data.cpu().numpy()[0]))

                        loss.backward()
                        optimizer.step()
                        output_cnt += 1
                        cnt = 1

                    cnt += 1
                    
                    if output_cnt % 100 ==0:
                        torch.save(unet.state_dict(),model_path+'/miccai_{}.pkl'.format(output_cnt))

                patch_n += 1

            cnt = 1
            patch_n = val_idx_start 
            
            DICE = 0.0
            dice_cnt = 0
            print('Validationn start...')
            while True:
                if patch_n >= all_num_patches: 
                    print('\nValidation done.\n')
                    break
                
                if patch_n >= val_idx_start and patch_n < val_idx_end:
                    m_p_path = glob(p_path+'/{}.mha'.format(patch_n))
                    m_l_path = glob(l_path+'/{}_l.mha'.format(patch_n))
                    if not m_p_path or not m_l_path: 
                        print('ERR file not exists')
                        break
                    val_batch = 32
                    x = np.zeros([val_batch, n_mode-1, patch_size[0], patch_size[1], patch_size[2]])
                    y = np.zeros([val_batch, n_channel, patch_size[0], patch_size[1], patch_size[2]])

                    p = io.imread(m_p_path[0], plugin='simpleitk').astype(float)
                    l = io.imread(m_l_path[0], plugin='simpleitk').astype(float) 
                    if l.sum() ==0:
                        patch_n += 1 
                        continue
                    for m in range(n_mode-1):
                        d1 = m*args.patch_size
                        d2 = (m+1)*args.patch_size
                        x[cnt-1,m] = p[d1:d2]
                    y[cnt-1,0] = l
                    print "#",
                    if cnt % val_batch == 0:
              
                        x_tensor = Variable(torch.from_numpy(x).float(), volatile=True).cuda()
                        y_tensor = Variable(torch.from_numpy(y).long()).cuda()

                        output = unet.forward(x_tensor)
                        output_arr = output.data.cpu().numpy()

                        output_cnt += 1
                        cnt = 1

                        # one hot encoding
                    
                        idx = output_arr[:,0]<output_arr[:,1]
                        idx = idx.reshape([val_batch, n_channel, patch_size[0], patch_size[1], patch_size[2]])
                        idx = idx.astype(np.int64)
                        print('\n - predict sum={}'.format(idx.sum()))
                        print(' - gt sum={}'.format(y.sum()))
                        
                        # DSC
                        denominator = idx.sum() + y.sum() 
                        print(' - denominator={}'.format(denominator))   
                        idx[idx==0] = -1
                        
                        intersection = y==idx
                        intersection = intersection.astype(np.int64)
                        dice = intersection.sum()*2 / denominator

                        print(' - tp={}'.format(intersection.sum()))
    
                        print(' - {} dice={}'.format(dice_cnt, dice))
                        file_dsc.write(' - {} dice={}\n'.format(dice_cnt, dice))
                        DICE += dice
                        
                        if minDice > dice:
                            minDice = dice
                        if maxDice < dice:
                            maxDice = dice
                        dice_cnt += 1
                    else:
                        cnt += 1
                patch_n += 1
            
            print('\n{} fold-validation : mean DSC={}'.format(it, DICE/dice_cnt))
            print('{} fold-validation : min~max DSC={}~{}\n'.format(it, minDice, maxDice))
            file_dsc.write('--------------> {} fold-validation : DSC={}\n'.format(it, DICE/dice_cnt))
            file_dsc.write('--------------> {} fold-validation : min~max DSC={}~{}\n'.format(it, minDice, maxDice))

        file_dsc.write("\n=================>>>> mean DSC Result={}\n".format(DICE/dice_cnt))
        file_dsc.write('=================>>>> min~max DSC={}~{}\n\n'.format(minDice, maxDice))

else:
    
    # test
    if True:
        test = Preprocessing(n_mode, n_class, n_patch, volume_size, patch_size, n4b, True, data_name, root, train_bool)
        unet = nn.DataParallel(UnetGenerator_3d(in_dim=n_mode-1,out_dim=out_dim,num_filter=16)).cuda()
        print('\nCreate volume for test...')
        test_p_path = test.test_preprocess()
        print('Done.\n')
    else:
        test_p_path = root + data_name + '/test_VOL'

    im_path = glob(test_p_path + '/**')
   
    # model loading
    models_path = glob(model_path+'/**')
    model = model_path+'/miccai_{}.pkl'.format(len(models_path)*100)

    if not os.path.isfile(model):
        print(model+' -> model not exists\n')
        model = model_path+'/miccai_{}.pkl'.format(len(models_path)*1000)
        if not os.path.isfile(model):
            print(model+' -> also this model not exists\n')

    unet.load_state_dict(torch.load(model))
    print(model+' -> model loading success.\n')
    for idx, im in enumerate(im_path):

        if not os.path.isfile(im):
            print(p+' -> not exists')
            continue
        print(im + ' -> try loading')
        volume = io.imread(im, plugin='simpleitk').astype(float)
        print('Volume loading success\n')
        output_prob = np.zeros([volume_size[0], volume_size[1], volume_size[2]])
        output_class = np.zeros([volume_size[0], volume_size[1], volume_size[2]])
        DICE = 0.0
        dice_cnt = 0
        strd = 4 # strides
        print('Patch prediction start...')

        tic = time.time()
        for z in range(strd,volume_size[0],strd):
            for y in range(strd,volume_size[1],strd):
                for x in range(strd,volume_size[2],strd):
                    d1 = z-int(patch_size[0]/2)
                    d2 = z+int(patch_size[0]/2)
                    h1 = y-int(patch_size[1]/2)
                    h2 = y+int(patch_size[1]/2)
                    w1 = x-int(patch_size[2]/2)
                    w2 = x+int(patch_size[2]/2)

                    if d1 < 0 or d2 > volume_size[0] or h1 < 0 or h2 > volume_size[1] or w1 < 0 or w2 > volume_size[2]:
                        continue
        
                    x = np.zeros([1, n_mode-1, patch_size[0], patch_size[1], patch_size[2]])
              
                    for m in range(n_mode-1):
                        x[0,m] = volume[m, d1:d2, h1:h2, w1:w2]

                    x_tensor = Variable(torch.from_numpy(x).float(), volatile=True).cuda()

                    output = unet.forward(x_tensor)
                    output_arr = output.data.cpu().numpy()

                    tp = output_arr[0,0]<output_arr[0,1]
                    tp = tp.astype(np.int64)
                    
                    output_prob[d1:d2, h1:h2, w1:w2] += output_arr[0,1].reshape([patch_size[0], patch_size[1], patch_size[2]])
                    output_class[d1:d2, h1:h2, w1:w2] += tp[0,0]
                    
            print(' -----> {}/{} success'.format(z,volume_size[0]))
        print('Done. (prediction elapsed: %.2fs)' % (time.time() - tic))
        # save
        thsd = pow(patch_size[0]/strd, 3)/4 

        print('threshold = {}\n'.format(thsd)) 
        print('min={}, max={}'.format(np.min(output_class),np.max(output_class)))
        print('min={}, max={}\n'.format(np.min(output_prob),np.max(output_prob)))
        output_class = output_class > thsd
        output_class = output_class.astype(np.int64)

        path = root + data_name + '/test_PNG/{}_{}_{}_{}'.format(idx,patch_size[0],n_patch,n_epoch)
        if not os.path.exists(path):
            os.makedirs(path)
        print(np.unique(output_class))
        print(output_prob.sum())
        # zero mean norm
        b, t = np.percentile(output_prob, (1,99))
        output_prob = np.clip(output_prob, b, t)
        output_prob = (output_prob - np.mean(output_prob)) / np.std(output_prob)
        
        if np.max(output_prob) !=0:
            output_prob /= np.max(output_prob)
        if np.min(output_prob) <= -1:
            output_prob /= abs(np.min(output_prob))
        print(output_prob.sum())
        i = 0
        for slice_prob, slice_class in zip(output_prob,output_class):
            io.imsave(path+'/{}_predict_prob.PNG'.format(i), slice_prob)
            io.imsave(path+'/{}_predict_class.PNG'.format(i), slice_class*10000)
            i += 1
        print('Volume saved.')
        # DSC
        label_path = glob(path + '/*label*')
        label_volume = np.zeros([volume_size[0], volume_size[1], volume_size[2]])

        for k, slice in enumerate(label_path):
            label_volume[k] = io.imread(slice, plugin='simpleitk').astype(float)
        label_volume[label_volume>0] = 1
        print('predict sum={}'.format(output_class.sum()))
        print('gt sum={}'.format(label_volume.sum()))

        denominator = output_class.sum() + label_volume.sum()
        print('denominator={}'.format(denominator))
        output_class[output_class==0] = -1

        intersection = label_volume==output_class
        intersection = intersection.astype(np.int64)
        dice = intersection.sum()*2 / denominator

        print('tp={}'.format(intersection.sum()))

        print('{} dice={}'.format(idx, dice))
        test_file_dsc.write('{} dice={}\n'.format(idx, dice))

 
