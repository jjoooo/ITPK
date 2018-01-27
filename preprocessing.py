import os
import numpy as np
from glob import glob
import random

# image data
from skimage import io
import SimpleITK as sitk

from patch_extraction import MakePatches

import warnings
warnings.filterwarnings("ignore")
    
try:
    xrange
except NameError:
    xrange = range

class Preprocessing(object):

    def __init__(self, n_mode, n_class, n_patch, volume_size, patch_size, n4, n4_apply, data_name, root, train_bl, dim):
        
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
        self.dim = dim

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
            for mode_ix in range(self.num_mode-1):
                normed_slices[mode_ix][slice_ix] =  self._normalize(self.slices_by_mode[mode_ix][slice_ix])
                if np.max(normed_slices[mode_ix][slice_ix]) != 0: # set values < 1
                        normed_slices[mode_ix][slice_ix] /= np.max(normed_slices[mode_ix][slice_ix])
                if np.min(normed_slices[mode_ix][slice_ix]) <= -1: # set values > -1
                        normed_slices[mode_ix][slice_ix] /= abs(np.min(normed_slices[mode_ix][slice_ix]))
            if not train_bl:
                l_path = self.root_path+'/test_label_PNG/{}'.format(idx)
                o_path = self.root_path+'/test_origin_PNG/{}'.format(idx)
                if not os.path.exists(l_path):
                    os.makedirs(l_path)
                if not os.path.exists(o_path):
                    os.makedirs(o_path)
                io.imsave(l_path+'/{}_label.PNG'.format(slice_ix), normed_slices[-1][slice_ix])
                io.imsave(o_path+'/{}_origin.PNG'.format(slice_ix), normed_slices[0][slice_ix])
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
        use_n4 = ''
        if self.n4bias:
            use_n4 = '_n4'
        p_path = self.root_path+'/patch/patch_{}'.format(self.patch_size[0])+use_n4
        l_path = self.root_path+'/label/label_{}'.format(self.patch_size[0])
        if not os.path.exists(p_path):
            os.makedirs(p_path)
        if not os.path.exists(l_path):
            os.makedirs(l_path)

        if len(glob(p_path+'/**')) > 1:
            return p_path, l_path, 0
        
        patch_n = 0
        len_patch = 0
        n_val = 2
        val_cnt = 0
        for idx, patient in enumerate(self.patients):

            if not self.volume2slices(patient):
                continue
            
            normed_slices = self.norm_slices(idx, self.training_bool)
            if val_cnt < n_val:
                val_str = '/test'
                print(patient)
            else:
                val_str = '/train'
            
            for i in range(self.num_class):
                if not os.path.exists(p_path+val_str):
                    os.makedirs(p_path+val_str)
                if not os.path.exists(l_path+val_str):
                    os.makedirs(l_path+val_str)

                if not os.path.exists(p_path+val_str+'/{}'.format(i)):
                    os.makedirs(p_path+val_str+'/{}'.format(i))
                if not os.path.exists(l_path+val_str+'/{}'.format(i)):
                    os.makedirs(l_path+val_str+'/{}'.format(i))

            # run patch_extraction
            pl = MakePatches(self.volume_size, self.patch_size ,self.num_mode, self.num_class, self.num_patch/len(self.patients), self.dim)

            if self.dim==2:
                l_p = pl.create_2Dpatches(normed_slices, p_path+val_str, l_path+val_str)
                len_patch += l_p
            else:
                l_p = pl.create_3Dpatches(normed_slices, p_path+val_str, l_path+val_str)
                len_patch += l_p
            
            val_cnt += 1
            print('-----------------------idx = {} & num of patches = {}'.format(idx, l_p))
        print('\n\nnum of all patch = {}'.format(len_patch))

        ''' 
        for c in range(num_class):
            for m in range(num_mode-1):
                p = random.choice(np.argwhere(np.asarray(self.center_labels) == c))
                p = p[0]
                fn = '/Users/jui/Downloads/Data/' + data_name + '_{}_{}.PNG'.format(m,c)
                io.imsave(fn, patches[m][p][7])
        '''
        return p_path, l_path, len_patch

    def test_preprocess(self):

        patch_n = 0
        test_path = self.root_path+'/test_VOL'
        if not os.path.exists(test_path):
            os.makedirs(test_path)
 
        use_n4 = ''
        if self.n4bias:
            use_n4 = '_n4'
        p_path = self.root_path+'/test_patch/patch_{}'.format(self.patch_size[0])+use_n4
        l_path = self.root_path+'/test_label/label_{}'.format(self.patch_size[0])
        if not os.path.exists(p_path):
            os.makedirs(p_path)
        if not os.path.exists(l_path):
            os.makedirs(l_path)

        for idx, patient in enumerate(self.patients):
            #if len(glob(test_path+'/{}.mha'.format(idx))) > 0: continue
            pn = self.volume_size[0]-int(self.patch_size[0]/2)
            
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
            
            sitk.WriteImage(sitk.GetImageFromArray(normed_slices), test_path+'/{}.mha'.format(idx))
        
            # run patch_extraction
            #pl = MakePatches(self.volume_size, self.patch_size ,self.num_mode, self.num_class, self.num_patch, self.dim)
            #pl.test_make_patch(normed_slices, p_path, idx)
            print('----------------idx = {} volume saved'.format(idx))
        return test_path

if __name__ == '__main__':
    pass
