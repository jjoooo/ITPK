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
parser.add_argument("--batch_size",type=int,default=64,help="batch size")
parser.add_argument("--num_gpu",type=int,default=8,help="number of gpus")
parser.add_argument("--patch_size",type=int,default=16,help="patch size")
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
n_class = 2
im_size = args.patch_size
n_mode = 4 # include gound truth
volume_size = (512,512,512)
patch_size = (im_size,im_size,im_size)
batch_size = args.batch_size
out_dim = 4

# Preprocessing
pp = Preprocessing(n_mode, n_class, 10000/n_class, volume_size, patch_size, False, 'MICCAI2008', root, True)

patches, labels, c_labels = pp.preprocess()

# Training (each mode)
x = Variable(torch.ones(batch_size,1,im_size,im_size,im_size)).cuda()
l = Variable(torch.zeros(batch_size,1,im_size,im_size,im_size)).cuda()
unet = []
for m in range(n_mode-1):
    unet[m] = nn.DataParallel(UnetGenerator_3d(in_dim=3,out_dim=out_dim,num_filter=4)).cuda()

for m in range(n_mode-1):
    cnt = 1
    output_cnt = 1
    for p, l in zip(patches, labels):
        x[cnt-1] = torch.from_numpy(p)
        l[cnt-1] = torch.from_numpy(l)
        if cnt % batch_size == 0:
            output = unet[m](x)
            #label = Variable(torch.zeros(batch_size,1,16,16,16).type_as(torch.LongTensor())).cuda()
            loss = loss_function(output,l)
            loss.backward()

            output_cnt += 1
            cnt = 1

        cnt += 1
    
        if output_cnt % 100 ==0:
            print('[MODE {}] - {}-------> loss : {}'.format(m, output_cnt, loss))
            