import os

import torch
from torch.autograd import Variable

from skimage import io
#from sklearn.preprocessing import minmax_scale

from glob import glob

import numpy as np
import time

from data_loader import Create_Batch
        
def testing(args, test_path, models, patient_idx):

    output_prob = np.zeros([args.volume_size, args.volume_size, args.volume_size])
    print('Patch segmentation test start...')

    tic = time.time()

    strd = 4
    padd = 100 

    mid = int(args.patch_size/2)
    test_imgs = Create_Batch(1, mid, args.n_mode-1, '')
    
    for z in range(padd,args.volume_size-strd,strd):
        for y in range(strd,args.volume_size-strd,strd):
            for x in range(strd,args.volume_size-strd,strd):
                p = glob(test_path+'/**/{}_{}_{}_*.PNG'.format(z,y,x))

                if not p: continue
                if len(p)>1: print('too many patches : '+ p[0])

                h1 = y-int(args.patch_size/2)
                h2 = y+int(args.patch_size/2)
                w1 = x-int(args.patch_size/2)
                w2 = x+int(args.patch_size/2)

                if h1 < 0 or h2 > volume_size[1] or w1 < 0 or w2 > volume_size[2]:
                    continue
    
                for m in range(n_mode-1):
                    if m==0: 
                        patch = volume[m, z, h1:h2, w1:w2]
                    else:
                        patch = np.concatenate((patch, volume[m, z, h1:h2, w1:w2]))

                im = io.imread(p[0], plugin='simpleitk').astype(float)
                patch = test_imgs.test_flip(im)

                patch = np.reshape(patch, (1, n_mode-1, args.patch_size, mid))
                patch = (patch-np.min(patch))/(np.max(patch)-np.min(patch))
                x1 = Variable(torch.from_numpy(patch[:,:,:mid]).float(), volatile=True).cuda()
                x2 = Variable(torch.from_numpy(patch[:,:,mid:]).float(), volatile=True).cuda()

                out_s = resnet_s.forward(x1)
                out_b = resnet_b.forward(x2)

                concat_out = torch.cat([out_s,out_b],dim=1)
                out = classifier.forward(concat_out)

                out_arr = out.data.cpu().numpy()  
                out_arr = out_arr[0][0]
                if out_arr > 0.000001:
                    print(out_arr)
                output_prob[z, h1:h2, w1:w2] += out_arr

        print(' -----> {}/{} success'.format(z,volume_size[0]))
    print('Done. (prediction elapsed: %.2fs)' % (time.time() - tic))s

    # will change adaptive thsd
    thsd = 0.4 #pow(patch_size[0]/strd, 3)/4 

    print('threshold = {}\n'.format(thsd)) 
    print('output_prob : min={}, max={}\n'.format(np.min(output_prob),np.max(output_prob)))

    path = root + data_name + '/test_result_PNG/{}_{}_{}_{}'.format(idx,patch_size[0],n_patch,n_epoch)
    if not os.path.exists(path):
        os.makedirs(path)
    
    # min-max scale
    output_prob = (output_prob-np.min(output_prob))/(np.max(output_prob)-np.min(output_prob))
    #output_prob[output_prob<thsd] = 0 

    print('output_prob : minmax-mean = {}'.format(np.mean(output_prob)))
    #output_prob = minmax_scale(output_prob)

    label_path = glob(root+data_name+'/test_label_PNG/{}/**'.format(idx))
    origin_path = glob(root+data_name+'/test_origin_PNG/{}/**'.format(idx))
    
    label_volume = np.zeros([volume_size[0], volume_size[1], volume_size[2]])
    origin_volume = np.zeros([volume_size[0], volume_size[1], volume_size[2]])
    
    for ix in range(len(origin_path)):
        label_volume[ix] = io.imread(root+data_name+'/test_label_PNG/{}/{}_label.PNG'.format(idx,ix), plugin='simpleitk').astype(long)
        origin_volume[ix] = io.imread(root+data_name+'/test_origin_PNG/{}/{}_origin.PNG'.format(idx,ix), plugin='simpleitk').astype(float)

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