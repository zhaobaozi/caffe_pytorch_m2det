"""
Copyright (c) 2017-present, starime.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.
"""
from __future__ import print_function
import os
import warnings
warnings.filterwarnings('ignore')
import time
import torch
import shutil
import argparse
from m2det import build_net
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from layers.functions import PriorBox
from data import detection_collate
from CC import Config
from utils.core import *
import os
import torch
import torch._utils

#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
parser = argparse.ArgumentParser(description='M2Det Training')
parser.add_argument('-c', '--config', default='configs/m2det512_vgg.py')
parser.add_argument('-d', '--dataset', default='VOC', help='VOC or COCO dataset')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-t', '--tensorboard', type=bool, default=False, help='Use tensorborad to show the Loss Graph')
args = parser.parse_args()
logger = set_logger(args.tensorboard)
global cfg
cfg = Config.fromfile(args.config)
pytorch_net = build_net('train', 
                size = cfg.model.input_size, # Only 320, 512, 704 and 800 are supported
                config = cfg.model.m2det_config)
#device = torch.device("cuda") # PyTorch v0.4.0
#pytorch_net = pytorch_net.to(device)
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import torchvision

from ConvertModel import ConvertModel_caffe
from ConvertModel import ConvertModel_ncnn

from ReplaceDenormals import ReplaceDenormals


""" Import your net structure here """

"""  ResNet  """



"""  Set empty path to use default weight initialization  """
# model_path = '../ModelFiles/ResNet/resnet50.pth'
model_path = '/home/ubuntu/zhq/M2Det/weights_allfinetuning8_5tum/M2Det_VOC_size512_netvgg16_epoch90.pth'
ModelDir = '/home/ubuntu/zhq/M2Det/model_ncnn_5tum/'

"""  Set to caffe or ncnn  """
dst = 'caffe'

if model_path != '':
    try:
        pretrained_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        model_dict = pytorch_net.state_dict()
            
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        pytorch_net.load_state_dict(model_dict)
    except AttributeError:
        pytorch_net = torch.load(model_path, map_location=lambda storage, loc: storage)
else:
    NetName = str(pytorch_net.__class__.__name__)
    if not os.path.exists(ModelDir + NetName):
        os.makedirs(ModelDir + NetName)
    print('Saving default weight initialization...')
    torch.save(pytorch_net.state_dict(), ModelDir + NetName + '/' + NetName + '.pth')

""" Replace denormal weight values(<1e-30), otherwise may increase forward time cost """
ReplaceDenormals(pytorch_net)

"""  Connnnnnnnvert!  """
print('Converting...')
InputShape=[1,3,512,512]
if dst == 'caffe':
    text_net, binary_weights = ConvertModel_caffe(pytorch_net.cuda(), InputShape, softmax=False)
elif dst == 'ncnn':
    text_net, binary_weights = ConvertModel_ncnn(pytorch_net.cuda(), InputShape, softmax=False)

"""  Save files  """
NetName = str(pytorch_net.__class__.__name__)
if not os.path.exists(ModelDir + NetName):
    os.makedirs(ModelDir + NetName)
print('Saving to ' + ModelDir + NetName)

if dst == 'caffe':
    import google.protobuf.text_format
    with open(ModelDir + NetName + '/' + NetName + '.prototxt', 'w') as f:
        f.write(google.protobuf.text_format.MessageToString(text_net))
    with open(ModelDir + NetName + '/' + NetName + '.caffemodel', 'wb') as f:
        f.write(binary_weights.SerializeToString())

elif dst == 'ncnn':
    import numpy as np
    with open(ModelDir + NetName + '/' + NetName + '.param', 'w') as f:
        f.write(text_net)
    with open(ModelDir + NetName + '/' + NetName + '.bin', 'w') as f:
        for weights in binary_weights:
            for blob in weights:
                blob_32f = blob.flatten().astype(np.float32)
                blob_32f.tofile(f)

print('Converting Done.')

