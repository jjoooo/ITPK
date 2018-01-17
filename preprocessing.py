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

    def __init__(self, n_mode, n_class, n_patch, volume_size, patch_size, n4, n4_apply, data_name, root, train_bl):
        
        self.num_mode = n_mode #4 #BRATS -> 5 # include ground truth
        self.num_class = n_class #2 #BRATS -> 5
        self.num_patch = n_patch 
        self.volume_size = volume_size #(512,512,512) #BRATS -> (155,240,240)
        self.patch_size = patch_size
        self.n4bias = n4
        self.n4bias_apply = n4_apply
        self.training_bool = train_bl
        self.data_name = data_name #'MICCAI2008' #BRATS -> 'BRATS2015'
        self.root_path = root + data_name

        self.path = ''
        self.ext = ''

        if self.data_name == 'BRATS2015':
            if train_bl:
                self.path = self.root_path + '/Original_Data/Training/HGG'
            else:
                self.path = self.root_path + '/Original_Data/Testing/HGG_LGG'
            self.ext = '.mha'
        elif self.data_name == 'MICCAI2008':
            if train_bl:
                self.path = self.root_path + '/training'
            else:
                self.path = self.root_path + '/test'
            self.ext = '.nhdr'

        self.patients = glob(self.path + '/**')

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

    def norm_slices(self, idx, train_bl):
        print('         -> Normalizing slices...')
        normed_slices = np.zeros((self.num_mode, self.volume_size[0], self.volume_size[1], self.volume_size[2]))
        for slice_ix in range(self.volume_size[0]):
            normed_slices[-1][slice_ix] = self.slices_by_mode[-1][slice_ix]
            if not train_bl:
                if not os.path.exists('./test/{}'.format(idx)):
                    os.makedirs('./test/{}'.format(idx))
                io.imsave('./test/{}/{}_label.PNG'.format(idx,slice_ix), normed_slices[-1][slice_ix])
                io.imsave('./test/{}/{}_origin.PNG'.format(idx,slice_ix), normed_slices[0][slice_ix])
            for mode_ix in range(self.num_mode-1):
                normed_slices[mode_ix][slice_ix] =  self._normalize(self.slices_by_mode[mode_ix][slice_ix])
                if np.max(normed_slices[mode_ix][slice_ix]) != 0: # set values < 1
                        normed_slices[mode_ix][slice_ix] /= np.max(normed_slices[mode_ix][slice_ix])
                if np.min(normed_slices[mode_ix][slice_ix]) <= -1: # set values > -1
                        normed_slices[mode_ix][slice_ix] /= abs(np.min(normed_slices[mode_ix][slice_ix]))

                # Test        
                # if slice_ix==76:
                #     io.imsave(self.root_path+'/miccai2008_{}_N4.PNG'.format(mode_ix), normed_slices[mode_ix][slice_ix])

        print('         -> Done.')
        return normed_slices

    def path_glob(self, data_name, path):
        if data_name == 'MICCAI2008':
            flair = glob(path + '/*FLAIR' + self.ext)
            t1 = glob(path + '/*T1' + self.ext)
            t1_n4 = glob(path + '/*T1*_n.mha')
            t2 = glob(path + '/*T2' + self.ext)
            gt = glob(path + '/*lesion' + self.ext)

        elif data_name == 'BRATS2015':
            flair = glob(path + '/*Flair*/*' + self.ext)
            t1 = glob(path + '/*T1*/*' + self.ext)
            t1_n4 = glob(path + '/*T1*/*_n.mha')
            t2 = glob(path + '/*_T2*/*' + self.ext)
            gt = glob(path + '/*OT*/*' + self.ext)

        return flair, t1, t1_n4, t2, gt

    def volume2slices(self, patient):
        print('         -> Loading scans...')

        mode = []
        # directories to each protocol (5 total)
        flair, t1s, t1_n4, t2, gt = self.path_glob(self.data_name, patient)
        t1 = [scan for scan in t1s if scan not in t1_n4]

        try:
            if not self.n4bias:
                if len(t1) > 1:
                    mode = [flair[0], t1[0], t1[1], t2[0], gt[0]]
                else:
                    mode = [flair[0], t1[0], t2[0], gt[0]]
            else:
                if self.n4bias_apply:
                    for im in t1:
                        self.n4itk_norm(im) # n4 normalize

                nm = '/*T1*_n.mha'
                if self.data_name == 'BRATS2015': nm = '/*T1*/'+nm

                t1_n4 = glob(patient + nm)

                if len(t1_n4) > 1:
                    mode = [flair[0], t1_n4[0], t1_n4[1], t2[0], gt[0]]
                else:
                    mode = [flair[0], t1_n4[0], t2[0], gt[0]]
        except IndexError:
            print('ERR(index err) : ' + patient)
            return False

        for scan_idx in range(len(mode)):
            self.slices_by_mode[scan_idx] = io.imread(mode[scan_idx], plugin='simpleitk').astype(float)
        print('         -> Done.')

        return True

    def n4itk_norm(self, path):
        img=sitk.ReadImage(path)
        img=sitk.Cast(img, sitk.sitkFloat32)
        img_mask=sitk.BinaryThreshold(img, 0, 0)
        
        print('             -> Applyling bias correction...')
        #corrector = sitk.N4BiasFieldCorrectionImageFilter()
        #corrected_img = corrector.Execute(img, img_mask)
        corrected_img = sitk.N4BiasFieldCorrection(img, img_mask, 0.001, [50,50,30,20])
        print('             -> Done.')
        sitk.WriteImage(corrected_img, path.replace(self.ext, '__n.mha'))

    def save_labels(self, labels):
        for label_idx in range(len(labels)):
            slices = io.imread(labels[label_idx], plugin = 'simpleitk')
            for slice_idx in range(len(slices)):
                io.imsave(self.path[:-3]+'Labels/{}_{}L.png'.format(label_idx, slice_idx), slices[slice_idx])

    def preprocess(self):

        p_path = self.root_path+'/patch/patch_{}_{}'.format(self.patch_size[0], self.num_patch)
        l_path = self.root_path+'/label/label_{}_{}'.format(self.patch_size[0], self.num_patch)
        if not os.path.exists(p_path):
            os.makedirs(p_path)
        if not os.path.exists(l_path):
            os.makedirs(l_path)
      
        if len(glob(p_path+'/**')) >= self.num_patch*0.8:
            print('         -> already training patches exist')
            return p_path, l_path
            
        patch_n = 0
        for idx, patient in enumerate(self.patients):

            if not self.volume2slices(patient):
                continue
            
            normed_slices = self.norm_slices(idx, self.training_bool)
            
            # run patch_extraction
            pl = Patches3d(self.volume_size, self.patch_size ,self.num_mode, self.num_class, self.num_patch/len(self.patients))
            pair_p, pair_l, self.center_labels = pl.make_patch(normed_slices, self.center_labels)
            print('-----------------------idx = {} & num of patches = {}'.format(idx, len(pair_p)))
            patient_n = 0
            for p,l in zip(pair_p, pair_l):
                temp = p_path+'/{}.mha'.format(patch_n)
                sitk.WriteImage(sitk.GetImageFromArray(p), temp)

                temp = l_path+'/{}_l.mha'.format(patch_n)
                sitk.WriteImage(sitk.GetImageFromArray(l), temp)

                patch_n += 1
     
        print('Complete.')

        ''' 
        for c in range(num_class):
            for m in range(num_mode-1):
                p = random.choice(np.argwhere(np.asarray(self.center_labels) == c))
                p = p[0]
                fn = '/Users/jui/Downloads/Data/' + data_name + '_{}_{}.PNG'.format(m,c)
                io.imsave(fn, patches[m][p][7])
        '''
        return p_path, l_path

    def test_preprocess(self):

        patch_n = 0
        test_path = self.root_path+'/patch/patch_{}_{}_test'.format(self.patch_size[0], self.num_patch)
        for idx, patient in enumerate(self.patients):
            p_path = test_path+'/{}'.format(idx)

            if not os.path.exists(p_path):
                os.makedirs(p_path)

            pn = self.volume_size[0]-int(self.patch_size[0]/2)
            if len(glob(p_path+'/**')) >= pn*pn*pn:
                print('         -> already test patches exist')
                continue

            flair, t1s, t1_n4, t2, gt = self.path_glob(self.data_name, patient)
            t1 = [scan for scan in t1s if scan not in t1_n4]

            try:
                if not self.n4bias:
                    if len(t1) > 1:
                        mode = [flair[0], t1[0], t1[1], t2[0], gt[0]]
                    else:
                        mode = [flair[0], t1[0], t2[0], gt[0]]
                else:
                    if self.n4bias_apply:
                        for im in t1:
                            self.n4itk_norm(im) # n4 normalize

                    nm = '/*T1*_n.mha'
                    if self.data_name == 'BRATS2015': nm = '/*T1*/'+nm

                    t1_n4 = glob(patient + nm)

                    if len(t1_n4) > 1:
                        mode = [flair[0], t1_n4[0], t1_n4[1], t2[0], gt[0]]
                    else:
                        mode = [flair[0], t1_n4[0], t2[0], gt[0]]
            except IndexError:
                print('ERR(index err) : ' + patient)
                return test_path

            for scan_idx in range(len(mode)):
                self.slices_by_mode[scan_idx] = io.imread(mode[scan_idx], plugin='simpleitk').astype(float)

            normed_slices = self.norm_slices(idx, self.training_bool)

            # run patch_extraction
            pl = Patches3d(self.volume_size, self.patch_size ,self.num_mode, self.num_class, self.num_patch)
            pl.test_make_patch(normed_slices, p_path, idx)
            print('------------------------------idx = {}'.format(idx))
        return test_path

if __name__ == '__main__':
    pass
