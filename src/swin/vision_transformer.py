##仅适用于224*224
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import copy
# import logging
# import torch
# import torch.nn as nn
# from src.swin.swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys
#
# logger = logging.getLogger(__name__)
#
#
# class SwinUnet(nn.Module):
#     def __init__(self, in_chans, img_size=224, num_classes=6, zero_head=False):
#         super(SwinUnet, self).__init__()
#         self.zero_head = zero_head
#         self.swin_unet = SwinTransformerSys(img_size=img_size,
#                                             patch_size=4,
#                                             in_chans=in_chans,
#                                             num_classes=num_classes)
#         # 全连接层，将特征映射为一个标量
#         self.fc = nn.Linear(301056, 1)  # 输入维度为301056
#
#     def forward(self, x):
#         if x.size(1) == 1:
#             # 如果输入为单通道，重复三次，变为三通道
#             x = x.repeat(1, 3, 1, 1)
#
#         # 获取 SwinTransformer 的输出特征 (已经是展平的)
#         x = self.swin_unet(x)
#
#         # 打印输出形状以调试
#         # print(f"Shape after swin_unet: {x.shape}")
#
#         # 直接通过全连接层得到标量输出
#         x = self.fc(x)
#         # 移除多余的维度
#         x = x.squeeze(-1)
#         return x
#
#     def load_from(self, config):
#         pretrained_path = config.MODEL.PRETRAIN_CKPT
#         if pretrained_path is not None:
#             print("pretrained_path:{}".format(pretrained_path))
#             device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#             pretrained_dict = torch.load(pretrained_path, map_location=device)
#
#             if "model" not in pretrained_dict:
#                 print("---start load pretrained model by splitting---")
#                 pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
#                 for k in list(pretrained_dict.keys()):
#                     if "output" in k:
#                         print("delete key:{}".format(k))
#                         del pretrained_dict[k]
#                 msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
#                 return
#
#             pretrained_dict = pretrained_dict['model']
#             print("---start load pretrained model of swin encoder---")
#
#             model_dict = self.swin_unet.state_dict()
#             full_dict = copy.deepcopy(pretrained_dict)
#             for k, v in pretrained_dict.items():
#                 if "layers." in k:
#                     current_layer_num = 3 - int(k[7:8])
#                     current_k = "layers_up." + str(current_layer_num) + k[8:]
#                     full_dict.update({current_k: v})
#
#             for k in list(full_dict.keys()):
#                 if k in model_dict:
#                     if full_dict[k].shape != model_dict[k].shape:
#                         del full_dict[k]
#
#             msg = self.swin_unet.load_state_dict(full_dict, strict=False)
#         else:
#             print("none pretrain")

####普遍适用

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

        # 动态计算 SwinTransformerSys 的输出特征维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_chans, img_size, img_size)
            dummy_output = self.swin_unet(dummy_input)
            self.feature_dim = dummy_output.shape[1]  # 获取特征维度

        # 全连接层，输入维度动态设置为 SwinTransformerSys 的输出特征维度
        self.fc = nn.Linear(self.feature_dim, 1)  # 输入维度为动态计算的 feature_dim

    def forward(self, x):
        if x.size(1) == 1:
            # 如果输入为单通道，重复三次，变为三通道
            x = x.repeat(1, 3, 1, 1)

        # 获取 SwinTransformer 的输出特征 (已经是展平的)
        x = self.swin_unet(x)

        # 打印输出形状以调试
        print(f"Shape after swin_unet: {x.shape}")

        # 直接通过全连接层得到标量输出
        x = self.fc(x)
        # 移除多余的维度
        x = x.squeeze(-1)
        return x

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)

            if "model" not in pretrained_dict:
                print("---start load pretrained model by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                return

            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained model of swin encoder---")

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
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
        else:
            print("none pretrain")