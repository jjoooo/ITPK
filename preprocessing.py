import os
import numpy as np
from glob import glob
import random

# image data
from skimage import io
import SimpleITK as sitk

from patch_extraction import Patches3d

import warnings
warnings.filterwarnings("ignore")
    
try:
    xrange
except NameError:
    xrange = range

class Preprocessing(object):

    def __init__(self, num_mode, num_class, num_patch, volume_size, patch_size, n4bias, data_name, root, training_bool):
        
        self.num_mode = num_mode #4 #BRATS -> 5 # include ground truth
        self.num_class = num_class #2 #BRATS -> 5
        self.num_patch = num_patch #10
        self.volume_size = volume_size #(512,512,512) #BRATS -> (155,240,240)
        self.patch_size = patch_size #(25,25,25)
        self.n4bias = n4bias #False
        self.training_bool = training_bool
        self.data_name = data_name #'MICCAI2008' #BRATS -> 'BRATS2015'
        self.root_path = root + data_name #'/Users/jui/Downloads/Data/' + data_name

        self.path = ''
        self.ext = ''

        if self.data_name == 'BRATS2015':
            self.path = self.root_path + '/Original_Data/Training/HGG'
            self.ext = '.mha'
        elif self.data_name == 'MICCAI2008':
            self.path = self.root_path + '/*train_Part*'
            self.ext = '.nhdr'

        self.patients = glob(self.path + '/**')

        self.pair_p = [[],[],[],[]]
        self.pair_l = []
        self.center_labels = []

        self.slices_by_mode = np.zeros((self.num_mode, self.volume_size[0], self.volume_size[1], self.volume_size[2]))
        
    def _normalize(self, slice):
        # remove outlier
        b, t = np.percentile(slice, (1,99))
        slice = np.clip(slice, 0, t)
        slice[slice<0] = 0
        if np.std(slice) == 0: 
            return slice
        else:
            # zero mean norm
            return (slice - np.mean(slice)) / np.std(slice)

    def norm_slices(self):
        print('Normalizing slices...')
        normed_slices = np.zeros((self.num_mode, self.volume_size[0], self.volume_size[1], self.volume_size[2]))
        for slice_ix in range(self.volume_size[0]):
            normed_slices[-1][slice_ix] = self.slices_by_mode[-1][slice_ix]
            for mode_ix in range(self.num_mode-1):
                normed_slices[mode_ix][slice_ix] =  self._normalize(self.slices_by_mode[mode_ix][slice_ix])
                if np.max(normed_slices[mode_ix][slice_ix]) != 0: # set values < 1
                        normed_slices[mode_ix][slice_ix] /= np.max(normed_slices[mode_ix][slice_ix])
                if np.min(normed_slices[mode_ix][slice_ix]) <= -1: # set values > -1
                        normed_slices[mode_ix][slice_ix] /= abs(np.min(normed_slices[mode_ix][slice_ix]))

        print('Done.')
        return normed_slices

    def path_glob(self, data_name, path):
        if data_name == 'MICCAI2008':
            flair = glob(path + '/*FLAIR' + self.ext)
            t1 = glob(path + '/*T1' + self.ext)
            t1_n4 = glob(path + '/*T1_n' + self.ext)
            t2 = glob(path + '/*T2' + self.ext)
            gt = glob(path + '/*lesion' + self.ext)

        elif data_name == 'BRATS2015':
            flair = glob(path + '/*Flair*/*' + self.ext)
            t1 = glob(path + '/*T1*/*' + self.ext)
            t1_n4 = glob(path + '/*T1*/*_n' + self.ext)
            t2 = glob(path + '/*_T2*/*' + self.ext)
            gt = glob(path + '/*OT*/*' + self.ext)

        return flair, t1, t1_n4, t2, gt

    def volume2slices(self, patient):
        print('Loading scans...')

        mode = []
        # directories to each protocol (5 total)
        flair, t1s, t1_n4, t2, gt = self.path_glob(self.data_name, patient)
        t1 = [scan for scan in t1s if scan not in t1_n4]

        if not self.n4bias:
            try:
                if len(t1) > 1:
                    mode = [flair[0], t1[0], t1[1], t2[0], gt[0]]
                else:
                    mode = [flair[0], t1[0], t2[0], gt[0]]
            except IndexError:
                print('ERR(index err) : '+t1)
                return False
        else:
            print('-> Applyling bias correction...')
            for im in t1:
                self.n4itk_norm(im) # n4 normalize
            t1_n4 = glob(patient + '/*T1_n' + self.ext)
            try:
                if len(t1_n4) > 1:
                    mode = [flair[0], t1_n4[0], t1_n4[1], t2[0], gt[0]]
                else:
                    mode = [flair[0], t1_n4[0], t2[0], gt[0]]
            except IndexError:
                print('ERR(index err) : '+t1_n4)
                return False
            print('-> Done.')

        for scan_idx in range(len(mode)):
            self.slices_by_mode[scan_idx] = io.imread(mode[scan_idx], plugin='simpleitk').astype(float)
        print('Done.')

        return True

    def n4itk_norm(self, path):
        img=sitk.ReadImage(path)
        img=sitk.Cast(img, sitk.sitkFloat32)
        img_mask=sitk.BinaryNot(sitk.BinaryThreshold(img, 0, 0))
        corrected_img = sitk.N4BiasFieldCorrection(img, img_mask, 0.001, [50,50,30,20])
        sitk.WriteImage(corrected_img, self.path.replace(self.ext, '_n' + self.ext))

    def preprocess(self):

        for idx, patient in enumerate(self.patients):

            if not self.volume2slices(patient):
                continue
            
            normed_slices = self.norm_slices()

            # run patch_extraction
            pl = Patches3d(self.volume_size, self.patch_size ,self.num_mode, self.num_class, self.num_patch)
            self.pair_p, self.pair_l, self.center_labels = pl.make_patch(normed_slices, self.pair_p, self.pair_l, self.center_labels)
            print('-----------------------------------------idx = {} & num of patches = {}'.format(idx, len(self.pair_l)))

        # run balancing
        # if self.training_bool:
        #     patches, labels, c_labels = pl.make_balance_patches(self.pair_p, self.pair_l, self.center_labels)

        # save example patches
        ''' 
        for c in range(num_class):
            for m in range(num_mode-1):
                p = random.choice(np.argwhere(np.asarray(c_labels) == c))
                p = p[0]
                fn = '/Users/jui/Downloads/Data/' + data_name + '_{}_{}.PNG'.format(m,c)
                io.imsave(fn, patches[m][p][7])
        '''

        print('Complete.')

        return self.pair_p, self.pair_l, self.center_labels

if __name__ == '__main__':
    pass