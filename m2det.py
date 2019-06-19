'''
This code is based on pytorch_ssd and RFBNet.
Details about the modules:
               TUM - Thinned U-shaped Module
               MLFPN - Multi-Level Feature Pyramid Network
               M2Det - Multi-level Multi-scale single-shot object Detector

Author:  Qijie Zhao (zhaoqijie@pku.edu.cn)
Finished Date:  01/17/2019

'''
import torch
#from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import warnings
warnings.filterwarnings('ignore')
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os,sys,time
from layers.nn_utils import *
from torch.nn import init as init
from utils.core import print_info
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
class M2Det(nn.Module):
    def __init__(self, phase, size, config = None):
        '''
        M2Det: Multi-level Multi-scale single-shot object Detector
        '''
        super(M2Det,self).__init__()
        self.phase = phase
        self.size = size
        self.init_params(config)
        print_info('===> Constructing M2Det model', ['yellow','bold'])
        self.construct_modules()

    def init_params(self, config=None): # Directly read the config
        assert config is not None, 'Error: no config'
        for key,value in config.items():
            if check_argu(key,value):
                setattr(self, key, value)

    def construct_modules(self,):
        # construct tums
        for i in range(self.num_levels):
            if i == 0:
                setattr(self,
                        'unet{}'.format(i+1),
                        TUM(first_level=True, 
                            input_planes=self.planes//2, 
                            is_smooth=self.smooth,
                            scales=self.num_scales,
                            side_channel=512)) #side channel isn't fixed.
            else:
                setattr(self,
                        'unet{}'.format(i+1),
                        TUM(first_level=False, 
                            input_planes=self.planes//2, 
                            is_smooth=self.smooth, 
                            scales=self.num_scales,
                            side_channel=self.planes))
        # construct base features
        if 'vgg' in self.net_family:
            self.base = nn.ModuleList(get_backbone(self.backbone))
            shallow_in, shallow_out = 512,256
            deep_in, deep_out = 1024,512
        elif 'res' in self.net_family: # Including ResNet series and ResNeXt series
            self.base = get_backbone(self.backbone)
            shallow_in, shallow_out = 512,256
            deep_in, deep_out = 2048,512
        self.reduce= BasicConv(shallow_in, shallow_out, kernel_size=3, stride=1, padding=1)
        self.up_reduce= BasicConv(deep_in, deep_out, kernel_size=1, stride=1)
        
        # construct others
        if self.phase == 'test':
            self.softmax = nn.Softmax()
        self.Norm2 = nn.BatchNorm2d(256*5)
        self.leach = nn.ModuleList([BasicConv(
                    deep_out+shallow_out,
                    self.planes//2,
                    kernel_size=(1,1),stride=(1,1))]*self.num_levels)

        # construct localization and recognition layers
        loc_2 = list()
        conf_2 = list()
        for i in range(self.num_scales):
            loc_2.append(nn.Conv2d(self.planes*self.num_levels,
                                       4 * 6, # 4 is coordinates, 6 is anchors for each pixels,
                                       3, 1, 1))
            conf_2.append(nn.Conv2d(self.planes*self.num_levels,
                                       self.num_classes * 6, #6 is anchors for each pixels,
                                       3, 1, 1))
        self.loc2 = nn.ModuleList(loc_2)
        self.conf2 = nn.ModuleList(conf_2)
        self.deconv1 = nn.Sequential()
        self.deconv1.add_module('{}'.format(len(self.deconv1)), nn.ConvTranspose2d(512, 512, 2, 2, 0))
        # construct SFAM module
        if self.sfam:
            self.sfam_module = SFAM(self.planes, self.num_levels, self.num_scales, compress_ratio=16)
    
    def forward(self,x):
        loc2,conf2 = list(),list()
        base_feats = list()
        if 'vgg' in self.net_family:
            for k in range(len(self.base)):
                #print('k:',k)
                x = self.base[k](x)
          
                #print('base_out',self.base_out)
                if k in self.base_out:
                    base_feats.append(x)
        elif 'res' in self.net_family:
            base_feats = self.base(x, self.base_out)
       # print self.reduce(base_feats[0]).shape
       # print self.up_reduce(base_feats[1]).shape
        base_feature = torch.cat((self.reduce(base_feats[0]), self.deconv1(self.up_reduce(base_feats[1]))), 1)
        # tum_outs is the multi-level multi-scale feature
        tum_outs = [getattr(self, 'unet{}'.format(1))(self.leach[0](base_feature), 'none')]
        #print('tum_outs',tum_outs)
        for i in range(1,self.num_levels,1):
            tum_outs.append(
                    getattr(self, 'unet{}'.format(i+1))(
                        self.leach[i](base_feature), tum_outs[i-1][-1]
                        )
                    )
        # concat with same scales
        sources = [torch.cat([_fx[i-1] for _fx in tum_outs],1) for i in range(self.num_scales, 0, -1)]
        
        # forward_sfam
        if self.sfam:
            sources = self.sfam_module(sources)
        sources[0] = self.Norm2(sources[0])
        
        for (x,l,c) in zip(sources, self.loc2, self.conf2):
            loc2.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf2.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc2 = torch.cat([o.view(o.size(0), -1) for o in loc2], 1)
        conf2 = torch.cat([o.view(o.size(0), -1) for o in conf2], 1)

        if self.phase == "test":
            output = (
                loc2.view(loc2.size(0), -1, 4),                   # loc preds
                self.softmax(conf2.view(-1, self.num_classes)),  # conf preds
            )
        else:
            output = (
                loc2.view(loc2.size(0), -1, 4),
                conf2.view(conf2.size(0), -1, self.num_classes),
            )
        return output

    def init_model(self, base_model_path):
        if self.backbone == 'vgg16':
            if isinstance(base_model_path, str):
                base_weights = torch.load(base_model_path)
                print_info('Loading base network...')
                self.base.load_state_dict(base_weights)
        elif 'res' in self.backbone:
            pass # pretrained seresnet models are initially loaded when defining them.
        
        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.kaiming_normal_(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0
        
        print_info('Initializing weights for [tums, reduce, up_reduce, leach, loc, conf]...')
        for i in range(self.num_levels):
            getattr(self,'unet{}'.format(i+1)).apply(weights_init)
        self.reduce.apply(weights_init)
        self.up_reduce.apply(weights_init)
        self.leach.apply(weights_init)
        self.loc2.apply(weights_init)
        self.conf2.apply(weights_init)

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print_info('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file))
            print_info('Finished!')
        else:
            print_info('Sorry only .pth and .pkl files supported.')

def build_net(phase='train', size=320, config = None):
    if not phase in ['test','train']:
        raise ValueError("Error: Phase not recognized")

    if not size in [320, 512, 704, 800]:
        raise NotImplementedError("Error: Sorry only M2Det320,M2Det512 M2Det704 or M2Det800 are supported!")
   # M2Det(phase, size, config)
   # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
   # model = M2Det(phase,size,config).to(device)
   # summary(model, (3, 224, 224))
    return M2Det(phase, size, config)
