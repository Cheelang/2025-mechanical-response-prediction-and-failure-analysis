# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import torch
import torch.nn as nn
from src.swin.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys

logger = logging.getLogger(__name__)


class SwinUnet(nn.Module):
    def __init__(self, in_chans, img_size=224, num_classes=6, zero_head=False):
        super(SwinUnet, self).__init__()
        self.zero_head = zero_head
        self.swin_unet = SwinTransformerSys(img_size=img_size,
                                            patch_size=4,
                                            in_chans=in_chans,
                                            num_classes=num_classes)
        # 新增全局平均池化层
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # 新增全连接层，将特征映射到标量
        self.fc = nn.Linear(self.swin_unet.num_features_up, 1)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.swin_unet(x)
        
        # 应用全局平均池化，将输出从 BxCxHxW 转换为 BxC
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten操作，得到 BxC
        # 通过全连接层得到标量输出
        x = self.fc(x)
        
        return x

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location = device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict = False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        # print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict = False)
            # print(msg)
        else:
            print("none pretrain")


