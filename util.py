import os
import numpy as np
from glob import glob

import torch
import torch.nn as nn
from resnet2d import Resnet, Classifier

# exist models
def pretr_model_loading(model_path, model):  
    
    # pretrain epoch
    print('pretrained s model loading: '+model_path)
    model.load_state_dict(torch.load(model_path))

    return model

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

    models_path = []
    model_idx = 1
    for m in range(len(models)):
        models_path.append(glob(path+'/*_{}_*.pkl'.format(m)))
        if models_path[m]:
            model_path = path + '/miccai_{}_{}.pkl'.format(m, len(models_path[m]))
            models[m] = pretr_model_loading(model_path, models[m])
            model_idx = len(models_path[m]) + 1

    return models, model_idx, path