import numpy as np
import random
import os
from glob import glob
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
from numpy.lib import stride_tricks

import warnings
warnings.filterwarnings("ignore")
    
try:
    xrange
except NameError:
    xrange = range

class Patches3d(object):
    def __init__(self, volume_size, patch_size, num_mode, num_class, num_patch):
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.d = self.patch_size[0]
        self.h = self.patch_size[1]
        self.w = self.patch_size[2]
        self.num_mode = num_mode
        self.num_class = num_class
        self.num_patch = num_patch
        
        self.d1 = int(self.d/2)
        self.d2 = self.volume_size[0] - int(self.d/2) -1
        self.h1 = int(self.h/2)
        self.h2 = self.volume_size[1] - int(self.h/2) -1
        self.w1 = int(self.w/2)
        self.w2 = self.volume_size[2] - int(self.w/2) -1

    def _create_patches(self, volume):

        volume_l = volume[-1]
        np.delete(volume, -1, 0)

        c_l = []
        for c in range(self.num_class):
            c_l.append(np.argwhere(volume_l == c))
            if self.num_patch > len(c_l[c]):
                self.num_patch = len(c_l[c])

        patches, labels, center_l = [[],[],[],[]], [], []
        for c in range(self.num_class):
        
            # random patch each class
            cnt = 0

            while cnt < self.num_patch:
                l_idx = random.choice(c_l[c])
              
                d1 = l_idx[0]-int(self.d/2)
                d2 = l_idx[0]+int(self.d/2)
                h1 = l_idx[1]-int(self.h/2)
                h2 = l_idx[1]+int(self.h/2)
                w1 = l_idx[2]-int(self.w/2)
                w2 = l_idx[2]+int(self.w/2)

                if d1 < 0 or d2 > self.volume_size[0] or h1 < 0 or h2 > self.volume_size[1] or w1 < 0 or w2 > self.volume_size[2]:
                    continue

                label = volume_l[d1:d2, h1:h2, w1:w2]
                cl = volume_l[l_idx[0],l_idx[1],l_idx[2]]

                # label filtering
                if not self._labels_filtering(label, cl):
                    continue

                # patch filtering
                bool_p = True
                for m in range(self.num_mode-1):
                    patch = volume[m, d1:d2, h1:h2, w1:w2]
                    if not self._patch_filtering(patch):
                        bool_p = False
                        break
                if not bool_p:
                    continue

                for m in range(self.num_mode-1):
                    patch = volume[m, d1:d2, h1:h2, w1:w2]
                    patches[m].append(patch)
                labels.append(label)
                center_l += [cl]
                cnt += 1
                    
        return patches, labels, center_l

    def _labels_filtering(self, label, cl):

        # label filtering
        if len(np.argwhere(label == cl)) < 10:
            return False
        
        # 80th entropy percentile
        m_ent = 0
        ent = np.ndarray([self.h,self.w])
        for i in range(self.d):
            l_ent = entropy(label[i].astype(int), disk(self.h))
            if m_ent < np.mean(l_ent):
                m_ent = np.mean(l_ent)
                ent = l_ent

        top_ent = np.percentile(ent, 90)

        # if 80th entropy percentile = 0
        if top_ent == 0:
            return False   

        return True

    def _patch_filtering(self, patch):

        # any patch is too small 
        if patch.shape != self.patch_size:
            return False

        # 80th percentile
        t = np.percentile(patch, 90)

        # if 80th percentile = 0
        if t == -1:
            return False

        return True

    def make_patch(self, volume, center_labels):
            
        patches, labels, center_l = self._create_patches(volume)
        center_labels += center_l

        return patches, labels, center_labels

if __name__ == '__main__':
    pass
