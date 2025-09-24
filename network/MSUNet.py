# from MSUNet.py
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging

import torch
import torch.nn as nn
from .model_parts import MSUNetSys

logger = logging.getLogger(__name__)

class MSUNet(nn.Module):
    def __init__(self, 
                 config, 
                 img_size = 1024, 
                 num_classes = 1, 
                 zero_head = False, 
                 vis = False,):
        super(MSUNet, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.ms_unet = MSUNetSys(img_size = config.DATA.IMG_SIZE,               # image size 
                                patch_size = config.MODEL.SWIN.PATCH_SIZE,      # patch size (4x4)
                                in_chans = config.MODEL.SWIN.IN_CHANS,          # number of input channels (3)
                                num_classes = self.num_classes,                 # number of classes
                                embed_dim = config.MODEL.SWIN.EMBED_DIM,        # dimension after path-embeding (128)
                                depths = config.MODEL.SWIN.DEPTHS,              # how many transformer blocks [2, 2, 18, 2]
                                num_heads = config.MODEL.SWIN.NUM_HEADS,        # number of attention heads [4, 8, 16, 32]
                                window_size = config.MODEL.SWIN.WINDOW_SIZE,    # self-attention-window-size 7x7
                                mlp_ratio = config.MODEL.SWIN.MLP_RATIO,        # indicates how much this intermediate layer is inflated.
                                qkv_bias = config.MODEL.SWIN.QKV_BIAS,          # qkv with bias (True)
                                qk_scale = config.MODEL.SWIN.QK_SCALE,          # overwritting if diffrent scale wanted 
                                drop_rate = config.MODEL.DROP_RATE,             # dropout MLP or features
                                drop_path_rate = config.MODEL.DROP_PATH_RATE,   # transformer blocks can be sciped (reduce overfitting)
                                ape = config.MODEL.SWIN.APE,                    # absolute position embedding (False for swin)
                                patch_norm = config.MODEL.SWIN.PATCH_NORM,      # if after patch-embedding a layernorm (True)
                                use_checkpoint = config.TRAIN.USE_CHECKPOINT    # activating gradient checkpoint (saves GPU-memory)
                               )                      

    def forward(self, x):
        if x.size()[1] != 3:
            msg = f"Expected 3 channels, but got {x.size(1)}"
            logger.error(msg)
            raise ValueError(msg)
        return self.ms_unet(x)
    
    def freeze_encoder(self, freeze):
        self.ms_unet.freeze_encoder(freeze)

    def unfreeze_encoder(self, layer_num):
        self.ms_unet.unfreeze_encoder(layer_num)

    # for loading pretrained weights
    def load_segface_weights(self, config):
        # pretrained_path
        pretrained_path = config.MODEL.PRETRAIN_SEGFACE

        if pretrained_path is not None:
            print("pretrained_path:{}\n".format(pretrained_path))
            # cpu or cuda
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            ckpt = torch.load(pretrained_path, map_location=device)

            pretrained_dict = ckpt["state_dict_backbone"]

            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.ms_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.ms_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.ms_unet.load_state_dict(full_dict, strict=False)
            print("Finished loading pretrained weights")
            # print(msg)
        else:
            print("none pretrain")



    def load_segface_weight(self, config):
        # cpu or cuda
        pretrained_path = config.MODEL.PRETRAIN_SEGFACE
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        segface_dict = torch.load(pretrained_path, map_location = device)
        segface_dict = segface_dict["state_dict_backbone"]
        nums = list(range(18))
        new_state_dict = {}
        skip = False

        for k, v in segface_dict.items():
            new_k = k
            skip = False
            if k.startswith("backbone"):
                if k.startswith("backbone.0.0.0."):
                    new_k = k.replace("backbone.0.0.0", "patch_embed.proj")
                elif k.startswith("backbone.0.0.2."):
                    new_k = k.replace("backbone.0.0.2", "patch_embed.norm")
                elif k.startswith("backbone.0.1.0."):
                    new_k = k.replace("backbone.0.1.0", "layers.0.blocks.0")
                elif k.startswith("backbone.0.1.1."):
                    new_k = k.replace("backbone.0.1.1", "layers.0.blocks.1")
                elif k.startswith("backbone.0.2."):
                    new_k = k.replace("backbone.0.2", "layers.0.downsample")
                elif k.startswith("backbone.0.3.0."):
                    new_k = k.replace("backbone.0.3.0", "layers.1.blocks.0")
                elif k.startswith("backbone.0.3.1."):
                    new_k = k.replace("backbone.0.3.1", "layers.1.blocks.1")
                elif k.startswith("backbone.0.4."):
                    new_k = k.replace("backbone.0.4", "layers.1.downsample")
                elif k.startswith("backbone.0.5."):
                    for i in nums: # 0 to 17
                        segface_str = "backbone.0.5." + str(i)
                        msunet_str = "layers.2.blocks." + str(i)
                        if k.startswith(segface_str):
                            new_k = k.replace(segface_str, msunet_str)
                            break
                elif k.startswith("backbone.0.6."):
                    new_k = k.replace("backbone.0.6", "layers.2.downsample")
                elif k.startswith("backbone.0.7.0."):
                    new_k = k.replace("backbone.0.7.0", "layers.3.blocks.0")
                elif k.startswith("backbone.0.7.1."):
                    new_k = k.replace("backbone.0.7.1", "layers.3.blocks.1")
                elif k.startswith("backbone.1."):
                    skip = True
                else:
                    msg = f"Key {k} not found in dictionary!!"
                    logger.error(msg)
                    raise ValueError(msg)
                
                if skip == False:
                    if new_k == k:
                        msg = f"Key {k} not replaced!"
                        logger.error(msg)
                        raise ValueError(msg)
                
                    new_state_dict[new_k] = v

        model_dict = self.ms_unet.state_dict()

        for k in list(new_state_dict.keys()):
                if k in model_dict:
                    if new_state_dict[k].shape != model_dict[k].shape:
                        msg = f"Key {k} does not match the dictionary of MSUNet!"
                        logger.error(msg)
                        raise ValueError(msg)

        msg = self.ms_unet.load_state_dict(new_state_dict, strict=False)
        #print(msg)
        print("End of the pretrained copying process")

"""
    # for loading pretrained weights
    def load_from(self, config):
        # pretrained_path
        pretrained_path = config.MODEL.PRETRAIN_CKPT

        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            # cpu or cuda
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)

            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.ms_unet.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.ms_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.ms_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
"""

    
 