import os
import numpy as np
from glob import glob
import random

# image data
from skimage import io, color, img_as_float
from skimage.exposure import adjust_gamma
from sklearn.preprocessing import minmax_scale
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
from unet3d import UnetGenerator_3d, loss_function
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_idx",type=int,default=0)
parser.add_argument("--n_epoch",type=int,default=100)
parser.add_argument("--patch_size",type=int,default=32)
parser.add_argument("--n_patch",type=int,default=10000)
parser.add_argument("--batch_size",type=int,default=256)
parser.add_argument("--root",type=str,default='/Users/jui/Downloads/Data/')
parser.add_argument("--data_name",type=str,default='MICCAI2008')
parser.add_argument("--n_class",type=int,default=2)
parser.add_argument("--n_mode",type=int,default=4)
parser.add_argument("--volume_size",type=int,default=512)
parser.add_argument("--learning_rate",type=float,default=0.0002)
parser.add_argument("--fold_val",type=int,default=5)
parser.add_argument("--train_bl",type=int,default=1)
parser.add_argument("--tr_dim",type=int,default=2)
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

dim = args.tr_dim

if dim == 2:
    patch_size = (args.patch_size, args.patch_size)
else:
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
print('patch dimension = {}'.format(dim))
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
if train_bool==0:
    test_file_dsc = open('./dsc/TEST_lr{}_bs{}_ps{}_np{}_{}fold_DSC'.format(lr,batch_size,patch_size[0],n_patch,fold_val), 'w')

model_path = './model/model_ps{}_np{}_lr{}_{}fold'.format(args.patch_size, n_patch, lr, fold_val)

if train_bool:
    # Preprocessing
    tr = Preprocessing(n_mode, n_class, n_patch, volume_size, patch_size, n4b, True, data_name, root, train_bool, dim)

    print('\nCreate patch for training...')
    p_path, l_path, all_len = tr.preprocess()
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
        
        # pretrain epoch
        e = (output_cnt-1) / ((n_patch*(fold_val-1)/fold_val)/batch_size)
        n_epoch = n_epoch - int(e)
        if n_epoch < 1: n_epoch = 1
        print('pretrain epoch = {} -> edit n_epoch\nn_epoch = {}'.format(int(e),n_epoch))
        print('pretrained model loading: '+md_path)
        unet.load_state_dict(torch.load(md_path))

    n_patient = len(glob(p_path+'/**'))
    tr_num = int(n_patient*(float(fold_val-1)/float(fold_val)))
    val_num = n_patient - tr_num
    print('{} fold-validation : tr={}, test={}, in {}'.format(fold_val, tr_num, n_patient-tr_num, n_patient))

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
                if patch_n > n_patch:  
                    print('Training done.\n')
                    break

                while True:             
                    patient_idx = np.random.randint(n_patient)
                    if patient_idx >= val_idx_start and  patient_idx < val_idx_end:
                        continue
                    else: break

                patches = glob(p_path+'/{}/**'.format(patient_idx))
                labels = glob(l_path+'/{}/**'.format(patient_idx))
                
                odd = True
                if odd:
                    patch_idx = np.random.randint(int((len(patches)-1)/2))*2
                    odd = False
                else:
                    patch_idx = np.random.randint(int((len(patches)-1)/2))*2+1
                    odd = True
                    
                    m_p_path = glob(patches[patch_idx])
                    m_l_path = glob(labels[patch_idx])
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
                    
                    x = np.zeros([batch_size, n_mode-1, patch_size[0], patch_size[1], patch_size[2]])
                    y = np.zeros([batch_size, n_channel, patch_size[0], patch_size[1], patch_size[2]])

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
                    
                    if cnt % batch_size == 0:
              
                        x_tensor = Variable(torch.from_numpy(x).float(), volatile=True).cuda()
                        y_tensor = Variable(torch.from_numpy(y).long()).cuda()

                        output = unet.forward(x_tensor)
                        output_arr = output.data.cpu().numpy()

                        output_cnt += 1
                        cnt = 1

                        # one hot encoding
                    
                        idx = output_arr[:,0]<output_arr[:,1]
                        idx = idx.reshape([batch_size, n_channel, patch_size[0], patch_size[1], patch_size[2]])
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
                        file_dsc.write(' - {} dice={}\n--------------------------------------\n'.format(dice_cnt, dice))
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
            file_dsc.write('\n--------------> {} fold-validation : mean DSC={}\n'.format(it, DICE/dice_cnt))
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
    im_path.sort()
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
        strd = 8 # strides
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
                    output_arr = output_arr.astype(np.float)

                    tp = output_arr[0,0]<output_arr[0,1]
                    tp = tp.astype(np.int64)
                    
                    output_prob[d1:d2, h1:h2, w1:w2] += output_arr[0,1]
                    output_class[d1:d2, h1:h2, w1:w2] += tp[0,0]
                    
            print(' -----> {}/{} success'.format(z,volume_size[0]))
        print('Done. (prediction elapsed: %.2fs)' % (time.time() - tic))
        # save
        thsd = 1 #pow(patch_size[0]/strd, 3)/4 

        print('threshold = {}\n'.format(thsd)) 
        print('min={}, max={}'.format(np.min(output_class),np.max(output_class)))
        print('min={}, max={}\n'.format(np.min(output_prob),np.max(output_prob)))
        # output_class = output_class > thsd
        output_class = output_class.astype(np.int64)

        path = root + data_name + '/test_result_PNG/{}_{}_{}_{}'.format(idx,patch_size[0],n_patch,n_epoch)
        if not os.path.exists(path):
            os.makedirs(path)
        print('output_class = {}'.format(np.unique(output_class)))
        
        # remove outlier
        b, t = np.percentile(output_prob, (1,99))
        output_prob = np.clip(output_prob, 0, t)
        # min-max scale
        output_prob = minmax_scale(output_prob)
        print('output_prob sum = {}'.format(output_prob.sum()))

        label_path = glob(root+data_name+'/test_label_PNG/{}/**'.format(idx))
        origin_path = glob(root+data_name+'/test_origin_PNG/{}/**'.format(idx))

        label_volume = np.zeros([volume_size[0], volume_size[1], volume_size[2]])
        origin_volume = np.zeros([volume_size[0], volume_size[1], volume_size[2]])

        k=0
        for la, ori in zip(label_path, origin_path):
            label_volume[k] = io.imread(la, plugin='simpleitk').astype(int)
            origin_volume[k] = io.imread(ori, plugin='simpleitk').astype(int)
            k += 1
        label_volume[label_volume>0] = 1

        vol = img_as_float(origin_volume)
        vol = adjust_gamma(color.gray2rgb(vol), 0.5)
        print(vol.shape) 
        rgb_class = color.gray2rgb(output_prob)
        rgb_class = adjust_gamma(rgb_class, 0.5)
        red_add = [0.5, 0.1, 0.1]
        vol[output_prob>0.2] += red_add

        i = 0
        for slice_prob, slice_class, slice_rgb in zip(output_prob,output_class,vol):
            io.imsave(path+'/{}_predict_prob.PNG'.format(i), slice_prob)
            io.imsave(path+'/{}_predict_class.PNG'.format(i), slice_class*255)
            io.imsave(path+'/{}_predict_rgb_class.PNG'.format(i), slice_rgb)
            i += 1
        print('Volume saved.')

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

 
