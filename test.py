import os

import torch
from torch.autograd import Variable

from skimage import io
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import roc_auc_score

from glob import glob

import numpy as np
import time

        
def testing(args, test_batch, models, idx):

    output_prob = np.zeros([args.volume_size, args.volume_size, args.volume_size])
    print('Patch segmentation test start...')

    tic = time.time()


    print('\nValidation start...')
    resnet_s = models[0]
    resnet_b = models[1]
    classifier = models[2]
    
    for img,_,p in test_batch:
        
        coord = []
        for pp in p:
            z,y,x,etc = pp.split('_',3)
            coord.append([z,y,x])

        mid = int(args.patch_size/2)

        x1 = Variable(img[:,:,:mid]).cuda()
        x2 = Variable(img[:,:,mid:]).cuda()
    
        out_s = resnet_s.forward(x1)
        out_b = resnet_b.forward(x2)

        concat_out = torch.cat([out_s,out_b],dim=1)
        out = classifier.forward(concat_out)

        out_arr = out.data.cpu().numpy()  
        tar_arr = _.data.cpu().numpy()
        
        bc = 0
        for cd in coord:
            z = int(cd[0])
            y = int(cd[1])
            x = int(cd[2])

            h1 = y-int(args.patch_size/2)
            h2 = y+int(args.patch_size/2)
            w1 = x-int(args.patch_size/2)
            w2 = x+int(args.patch_size/2)

            if h1 < 0 or h2 > args.volume_size or w1 < 0 or w2 > args.volume_size:
                continue
            
            out_arr = out_arr[bc][0]
            
            output_prob[z, h1:h2, w1:w2] += out_arr
            bc += 1

        thsd = roc_auc_score(tar_arr, out_arr)
    print('Done. (prediction elapsed: %.2fs)' % (time.time() - tic))

    print('threshold = {}\n'.format(thsd)) 
    print('output_prob : min={}, max={}\n'.format(np.min(output_prob),np.max(output_prob)))

    path = root + data_name + '/test_result_PNG/{}_{}_{}_{}'.format(idx, args.patch_size, args.n_patch, args.n_epoch)
    if not os.path.exists(path):
        os.makedirs(path)
    
    # min-max scale
    output_prob[output_prob<thsd] = 0 

    output_prob = minmax_scale(output_prob)
    print('output_prob : minmax-mean = {}'.format(np.mean(output_prob)))

    label_path = glob(args.root+args.data_name+'/test_label_PNG/{}/**'.format(idx))
    origin_path = glob(args.root+args.data_name+'/test_origin_PNG/{}/**'.format(idx))
    
    label_volume = np.zeros([volume_size[0], volume_size[1], volume_size[2]])
    origin_volume = np.zeros([volume_size[0], volume_size[1], volume_size[2]])
    
    for ix in range(len(origin_path)):
        label_volume[ix] = io.imread(args.root+args.data_name+'/test_label_PNG/{}/{}_label.PNG'.format(idx,ix), plugin='simpleitk').astype(long)
        origin_volume[ix] = io.imread(args.root+args.data_name+'/test_origin_PNG/{}/{}_origin.PNG'.format(idx,ix), plugin='simpleitk').astype(float)

    vol = img_as_float(origin_volume)
    vol = (vol-np.min(vol))/(np.max(vol)-np.min(vol))
    vol = adjust_gamma(color.gray2rgb(vol), 0.5)

    red_mul = [1,0,0]
    rgb_class = color.gray2rgb(output_prob)
    rgb_class = red_mul * rgb_class
    rgb_class = adjust_gamma(rgb_class, 0.5)
    vol[output_prob>thsd] += rgb_class[output_prob>thsd]
    vol = (vol - np.mean(vol)) / np.std(vol)
    if np.max(vol) != 0: # set values < 1
        vol /= np.max(vol)
    if np.min(vol) <= -1: # set values > -1
        vol /= abs(np.min(vol))
    print('vol : min={},max={}'.format(np.min(vol),np.max(vol)))

    i = 0
    for slice_prob, slice_rgb in zip(output_prob,vol):
        io.imsave(path+'/{}_predict_prob.PNG'.format(i), slice_prob)
        io.imsave(path+'/{}_predict_rgb_class.PNG'.format(i), slice_rgb)
        i += 1
    print('Volume saved.')