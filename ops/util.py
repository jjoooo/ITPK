import os
import numpy as np
from glob import glob

import torch
import torch.nn as nn
from resnet2d import Resnet, Classifier


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
