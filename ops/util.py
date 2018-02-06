import os
import numpy as np
from glob import glob

import torch
import torch.nn as nn
from skimage import io

from resnet2d import Resnet, Classifier
import warnings
warnings.filterwarnings("ignore")


def init_model(args):
    # Model init
    resnet_s = nn.DataParallel(Resnet(args.n_mode-1)).cuda() # center patch model
    resnet_b = nn.DataParallel(Resnet(args.n_mode-1)).cuda() # downsampling patch model
    classifier = Classifier(args.batch_size).cuda()
    models = [resnet_s, resnet_b, classifier]

    # If is_exists Pretrained model: loading
    if not os.path.exists('./trained_model'):
        os.makedirs('./trained_model')

    path = './trained_model/model_lr{}_ps{}'.format(args.learning_rate, args.patch_size)
    if not os.path.exists(path):
        os.makedirs(path)

    model_idx = 1
    for m in range(len(models)):
        models_path = glob(path+'/miccai_{}.pkl'.format(m))

        # exist models
        if models_path:
            model_path = path+'/miccai_{}.pkl'.format(m)
            print('pretrained s model loading: '+model_path)
            models[m].load_state_dict(torch.load(model_path))

    return models, path



def dcm2png(path):
    patients = glob(path + '/**')
    
    for patient in patients:

        im_path = glob(patient + '/**/*.dcm')
        for p in im_path:
            fn = p[:-3]+'PNG'
            img = io.imread(p, plugin='simpleitk').astype(float)
            t = np.percentile(img, 99)
            img = np.clip(img, 0, t)
            img = (img - np.mean(img)) / np.std(img)
            if np.max(img) != 0: # set values < 1
                img /= np.max(img)
            if np.min(img) <= -1: # set values > -1
                img /= abs(np.min(img))

            io.imsave(fn, img[0])

if __name__ == "__main__":
    dcm2png('/Users/jui/Downloads/Data/YS/MS')

    
