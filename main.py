import os
import numpy as np
from glob import glob
import random

# image data
from skimage import io, color, img_as_float
from skimage.exposure import adjust_gamma
#from sklearn.preprocessing import minmax_scale
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
from resnet2d import Resnet, Classifier
from data_loader import Create_Batch
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_idx",type=int,default=0)
parser.add_argument("--n_epoch",type=int,default=100)
parser.add_argument("--patch_size",type=int,default=32)
parser.add_argument("--n_patch",type=int,default=200)
parser.add_argument("--batch_size",type=int,default=128)
parser.add_argument("--root",type=str,default='/mnt/disk1/data/MRI_Data/')
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
use_gpu = '{},{}'.format(args.gpu_idx,args.gpu_idx+1)
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
print('Training = {}'.format(train_bool))
print('----------------------------------------------')

if not os.path.exists('./loss'):
    os.makedirs('./loss')
if not os.path.exists('./acc'):
    os.makedirs('./acc')
if not os.path.exists('./model'):
    os.makedirs('./model')

# create file
file_loss = open('./loss/lr{}_bs{}_ps{}_np{}_ep{}_bce_loss'.format(lr,batch_size,patch_size[0],n_patch,n_epoch), 'w')
file_acc = open('./acc/lr{}_bs{}_ps{}_np{}_ep{}_acc'.format(lr,batch_size,patch_size[0],n_patch,n_epoch), 'w')
model_path = './model/model_ps{}_np{}_lr{}_ep{}'.format(args.patch_size, n_patch, lr, n_epoch)
if not os.path.exists(model_path):
    os.makedirs(model_path)

resnet_s = nn.DataParallel(Resnet(n_mode-1)).cuda()
resnet_b = nn.DataParallel(Resnet(n_mode-1)).cuda()
classifier = Classifier(batch_size).cuda()
if train_bool:
    # Preprocessing
    tr = Preprocessing(n_mode, n_class, n_patch, volume_size, patch_size, n4b, True, data_name, root, train_bool, dim)

    print('\nCreate patch for training...')
    p_path, l_path, all_len = tr.preprocess()
    print('Done.\n')

    #unet = nn.DataParallel(UnetGenerator_3d(in_dim=n_mode-1,out_dim=out_dim,num_filter=16)).cuda()
    
    model_cnt = 1
    optimizer_s = torch.optim.Adam(resnet_s.parameters(),lr=lr)
    optimizer_b = torch.optim.Adam(resnet_b.parameters(),lr=lr)
    loss_func = nn.BCEWithLogitsLoss()
    
    # exist models
    
    models_clf_path = glob(model_path+'/*_clf_*.pkl')
    models_s_path = glob(model_path+'/*_s_*.pkl')
    models_b_path = glob(model_path+'/*_b_*.pkl')
    if models_clf_path:
        md_clf_path = model_path+'/miccai_clf_{}.pkl'.format(len(models_clf_path))
        
        model_cnt = len(models_clf_path)+1
        
        # pretrain epoch
        print('pretrained s model loading: '+md_clf_path)
        classifier.load_state_dict(torch.load(md_clf_path))

    if models_s_path:
        md_s_path = model_path+'/miccai_s_{}.pkl'.format(len(models_s_path))
        
        model_cnt = len(models_s_path)+1
        
        # pretrain epoch
        print('pretrained s model loading: '+md_s_path)
        resnet_s.load_state_dict(torch.load(md_s_path))

    if models_b_path:
        md_b_path = model_path+'/miccai_b_{}.pkl'.format(len(models_b_path))
        
        model_cnt = len(models_b_path)+1
        
        # pretrain epoch
        print('pretrained b model loading: '+md_b_path)
        resnet_b.load_state_dict(torch.load(md_b_path))
    
    # Training data setting
    tr_bc = Create_Batch(batch_size, int(args.patch_size/2), n_mode-1, p_path+'/train')
    tr_batch = tr_bc.db_load()

    test_bc = Create_Batch(batch_size, int(args.patch_size/2), n_mode-1, p_path+'/test')
    test_batch = test_bc.db_load()

    cnt = 1
    for ep in range(n_epoch):
        
        for img,_ in tr_batch:
   
            optimizer_s.zero_grad()
            optimizer_b.zero_grad()
            
            mid = int(patch_size[0]/2)
            x1 = Variable(img[:,:,:mid]).cuda()
            x2 = Variable(img[:,:,mid:]).cuda()
            out_s = resnet_s.forward(x1)  
            out_b = resnet_b.forward(x2)

            concat_out = torch.cat([out_s,out_b],dim=1)
            out = classifier.forward(concat_out)

            target = Variable(_).float().cuda()
            target = target.view(batch_size,-1)
            loss = loss_func(out, target)

            file_loss.write('batch {} \t: loss {}\n'.format(cnt-1, loss.data.cpu().numpy()[0]))
            print('batch {} \t-------> loss {}'.format(cnt-1, loss.data.cpu().numpy()[0]))

            loss.backward()
            
            optimizer_s.step()
            optimizer_b.step()
            
            if cnt % 100 ==0:
                torch.save(classifier.state_dict(),model_path+'/miccai_clf_{}.pkl'.format(model_cnt))
                torch.save(resnet_s.state_dict(),model_path+'/miccai_s_{}.pkl'.format(model_cnt))
                torch.save(resnet_b.state_dict(),model_path+'/miccai_b_{}.pkl'.format(model_cnt))
                model_cnt += 1
            cnt += 1
        print('\nValidationn start...')
        trsd = 0.5
        ac = 0.0
        total = 0
        sum_out = 0
        for img,_ in test_batch:
            mid = int(patch_size[0]/2)
            mid = int(patch_size[0]/2)

            x1 = Variable(img[:,:,:mid]).cuda()
            x2 = Variable(img[:,:,mid:]).cuda()

            out_s = resnet_s.forward(x1)
            out_b = resnet_b.forward(x2)

            concat_out = torch.cat([out_s,out_b],dim=1)

            out = classifier.forward(concat_out)
            out = nn.Sigmoid()(out)
            out = out.view(batch_size,-1)
            for b in range(batch_size):
                out_val = out.data.cpu().numpy()[b,0]
                if out_val  > trsd:
                    ac += 1
                sum_out += out_val
                total += 1
        print('predict avg = {}%, accuracy = {}%'.format(sum_out/total*100, ac/total*100))
        file_loss.write('Epoch : predict avg = {}%, accuracy = {}%\n'.format(ep, sum_out/total*100, ac/total*100))
        print('Validationn done.\n') 

else:
    
    # test
    if True:
        test = Preprocessing(n_mode, n_class, n_patch, volume_size, patch_size, n4b, True, data_name, root, train_bool, dim)
        print('\nCreate volume for test...')
        test_p_path = test.test_preprocess()
        print('Done.\n')
    else:
        test_p_path = root + data_name + '/test_VOL'

    im_path = glob(test_p_path + '/**')
    im_path.sort()
    # model loading
    models_path = glob(model_path+'/**')
    model = model_path+'/miccai_{}.pkl'.format(len(models_path))
    if not os.path.isfile(model):
        print(model+' -> model not exists\n')
        model = model_path+'/miccai_{}.pkl'.format(len(models_path))
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
                    h1 = y-int(patch_size[1]/2)
                    h2 = y+int(patch_size[1]/2)
                    w1 = x-int(patch_size[2]/2)
                    w2 = x+int(patch_size[2]/2)

                    if h1 < 0 or h2 > volume_size[1] or w1 < 0 or w2 > volume_size[2]:
                        continue
        
                    for m in range(n_mode-1):
                        x[0,m] = volume[m, z, h1:h2, w1:w2]

                    x_tensor = Variable(torch.from_numpy(x).float(), volatile=True).cuda()

                    output = resnet_s.forward(x_tensor)
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
        output_prob = (output_prob-np.min(output_prob))/(np.max(output_prob)-np.min(output_prob))
        #output_prob = minmax_scale(output_prob)
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

 
