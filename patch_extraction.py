import numpy as np
import random
import os
from glob import glob
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
import SimpleITK as sitk

import warnings
warnings.filterwarnings("ignore")
    
try:
    xrange
except NameError:
    xrange = range

class MakePatches(object):
    def __init__(self, volume_size, patch_size, num_mode, num_class, num_patch, dim):
        self.volume_size = volume_size
        self.patch_size = patch_size
        if dim==3:
            self.d = self.patch_size[0]
            self.h = self.patch_size[1]
            self.w = self.patch_size[2]
        else: 
            self.h = self.patch_size[0]
            self.w = self.patch_size[1]

        self.num_mode = num_mode
        self.num_class = num_class
        self.num_patch = num_patch
        self.dim = dim
        
    def create_3Dpatches(self, volume, p_path, l_path, idx):

        volume_l = volume[-1]
        np.delete(volume, -1, 0)

        c_l = []
        min_c = self.num_patch
        for c in range(self.num_class):
            c_l.append(np.argwhere(volume_l == c))
            if c==0: min_c = len(c_l[c])
            else:
                if  min_c  > len(c_l[c]):
                    min_c  = len(c_l[c])
            print('class {} - {}'.format(c,len(c_l[c])))

        self.num_patch = int(min_c*2)
        print('num patch = {}'.format(self.num_patch))
    
        # random patch each class
        cnt = 0      
        l_n = 0
        cnt_bl = True  
        while cnt < self.num_patch and cnt_bl:
            
            for c in range(self.num_class):
                class_c_bl = True
                while class_c_bl:

                    if self.num_patch < len(c_l[c]):
                        l_idx = random.choice(c_l[c])
                    else:
                        l_idx = c_l[c][l_n]
                        l_n += 1
                        if l_n >= len(c_l[c]):
                            cnt_bl = False

                    d1 = l_idx[0]-int(self.d/2)
                    d2 = l_idx[0]+int(self.d/2)
                    h1 = l_idx[1]-int(self.h/2)
                    h2 = l_idx[1]+int(self.h/2)
                    w1 = l_idx[2]-int(self.w/2)
                    w2 = l_idx[2]+int(self.w/2)
                    if d1 < 0 or d2 > self.volume_size[0] or h1 < 0 or h2 > self.volume_size[1] or w1 < 0 or w2 > self.volume_size[2]:
                        continue

                    label = volume_l[d1:d2, h1:h2, w1:w2]
                
                    # label filtering
                    if not self._labels_filtering(label, c):
                        continue

                    # patch filtering
                    bool_p = True
                    for m in range(self.num_mode-1):
                        patch = volume[m, d1:d2, h1:h2, w1:w2]
                        if not self._patch_filtering(patch,c):
                            bool_p = False
                            break
                    if not bool_p:
                        continue
                
                    for m in range(self.num_mode-1):
                        patch = volume[m, d1:d2, h1:h2, w1:w2]

                        if m==0:
                            patch_mode = patch
                        else:
                            patch_mode = np.concatenate((patch_mode, patch))

                    temp = p_path+'/{}_{}.mha'.format(idx, cnt)
                    sitk.WriteImage(sitk.GetImageFromArray(patch_mode), temp)

                    temp = l_path+'/{}_{}_l.mha'.format(idx, cnt)
                    sitk.WriteImage(sitk.GetImageFromArray(label), temp)
                    cnt += 1
                    class_c_bl = False
                    
        return self.num_patch

    def create_2Dpatches(self, volume, p_path, l_path):

        volume_l = volume[-1]
        np.delete(volume, -1, 0)

        c_l = []
        min_c = self.num_patch/2
        for c in range(self.num_class):
            c_l.append(np.argwhere(volume_l == c))
            if  min_c  > len(c_l[c]):
                min_c  = len(c_l[c])
            print('class {} - {}'.format(c,len(c_l[c])))

        self.num_patch = min_c*2
        print('num patch = {}'.format(self.num_patch))

        # random patch each class
        cnt = 0
        l_n = 0
        cnt_bl = True
        while cnt < self.num_patch and cnt_bl:
            
            for c in range(self.num_class):
                class_c_bl = True
                while class_c_bl:
                    l_idx = random.choice(c_l[c])
                    if self.num_patch == len(c_l[c]):
                        l_idx = c_l[c][l_n]
                        l_n += 1
                        if l_n >= len(c_l[c]):
                            cnt_bl = False
                    
                    h1 = l_idx[1]-int(self.h/2)
                    h2 = l_idx[1]+int(self.h/2)
                    w1 = l_idx[2]-int(self.w/2)
                    w2 = l_idx[2]+int(self.w/2)
                    if h1 < 0 or h2 > self.volume_size[1] or w1 < 0 or w2 > self.volume_size[2]:
                        continue

                    label = volume_l[l_idx[0], h1:h2, w1:w2]

                    # label filtering
                    if not self._labels_filtering(label, c):
                        continue

                    # patch filtering
                    bool_p = True
                    for m in range(self.num_mode-1):
                        patch = volume[m, l_idx[0], h1:h2, w1:w2]
                        if not self._patch_filtering(patch,c):
                            bool_p = False
                            break
                    if not bool_p:
                        continue 
                    
                    for m in range(self.num_mode-1):
                        if m==0:
                            patches = volume[m, l_idx[0], h1:h2, w1:w2]
                        else:
                            patches = np.concatenate((patches, volume[m, l_idx[0], h1:h2, w1:w2]))

                    if np.max(patches) != 0: # set values < 1
                        patches /= np.max(patches)
                    if np.min(patches) <= -1: # set values > -1
                        patches /= abs(np.min(patches))

                    temp = p_path+'/{}/{}.PNG'.format(c,cnt)
                    io.imsave(temp, patches)

                    temp = l_path+'/{}/{}_l.PNG'.format(c,cnt)
                    #io.imsave(temp, label)
                    cnt += 1
                    class_c_bl = False

        return cnt

    def _labels_filtering(self, label, c):

        # label filtering
        if c==0:
            if len(np.unique(label)) != 1:
                return False

        if len(np.argwhere(label == c)) < 10:
            return False
    
        '''
        # 80th entropy percentile
        m_ent = 0
        ent = np.zeros([self.h,self.w])
        for i in range(self.d):
            l_ent = entropy(label[i].astype(int), disk(self.h))
            if m_ent < np.mean(l_ent):
                m_ent = np.mean(l_ent)
                ent = l_ent

        top_ent = np.percentile(ent, 90)

        # if 80th entropy percentile = 0
        if top_ent == 0:
            return False   
        '''
        return True

    def _patch_filtering(self, patch, c):

        # any patch is too small 
        if patch.shape != self.patch_size:
            print('patch shape mismatch')
            return False

        # 80th percentile
        t = np.percentile(patch, 90)

        # if 80th percentile = 0
        if t == -1:
            print('percentile 90 == -1')
            return False
        
        # 80th entropy percentile
        if c==0:
            m_ent = 0
            ent = np.zeros([self.h,self.w])
            if self.dim==3:
                for i in range(self.d):
                    l_ent = entropy(patch[i].astype(int), disk(self.h))
                    if m_ent < np.mean(l_ent):
                        m_ent = np.mean(l_ent)
                        ent = l_ent
            else:
                l_ent = entropy(patch.astype(int), disk(self.h))
                if m_ent < np.mean(l_ent):
                    m_ent = np.mean(l_ent)
                    ent = l_ent
            top_ent = np.percentile(ent, 90)

            # if 80th entropy percentile = 0
            if top_ent == 0:
                return False
        return True

    def test_make_patch(self, volume, p_path, patient_idx):
        print('idx {} : Make patches...'.format(patient_idx))
        patches = []
        idx,d,h,w = volume.shape
        cnt = 0
        for z in range(d):
            for y in range(h):
                for x in range(w): 
                    h1 = y-int(self.h/2)
                    h2 = y+int(self.h/2)
                    w1 = x-int(self.w/2)
                    w2 = x+int(self.w/2)

                    if h1 < 0 or h2 > h or w1 < 0 or w2 > w:
                        continue
                    
                    for m in range(idx-1):
                        if m==0:
                            patches = volume[m, z, h1:h2, w1:w2]
                        else:
                            patches = np.concatenate((patches, volume[m, z, h1:h2, w1:w2]))

                    if np.max(patches) != 0: # set values < 1
                        patches /= np.max(patches)
                    if np.min(patches) <= -1: # set values > -1
                        patches /= abs(np.min(patches))
                    
                    temp = p_path+'/{}_{}_{}.PNG'.format(z, y, x)
                    io.imsave(temp, patches)

                    #temp = l_path+'/{}/{}_l.PNG'.format(c,cnt)
                    #io.imsave(temp, label)
                    #sitk.WriteImage(sitk.GetImageFromArray(patch_mode), temp)
                    cnt += 1
        print('idx {} : Done.'.format(patient_idx))
if __name__ == '__main__':
    pass
