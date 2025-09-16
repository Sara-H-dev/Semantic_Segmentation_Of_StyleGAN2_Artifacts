# -----------------------------------------------------------------------------
# This file is adapted from the SegFace project:
# https://github.com/kartik-3004/segface
# network/models/segface_celeb.py
#
# Original code is licensed under the MIT License:
# 
# Copyright (c) 2024 Kartik Narayan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from typing import Any, Optional, Tuple, Type
from torchvision.models import convnext_large, convnext_base, convnext_small, convnext_tiny, swin_b, swin_v2_b, swin_v2_s, swin_v2_t, mobilenet_v3_large, efficientnet_v2_m
import pdb
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from network.transformer import *
from network.tray.utils_models import *

class MLPSegFace(nn.Module):
    """
    Multi-Layer Perceptron
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        # For all except the last one: RELU
        # For the last one: linear
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        # Optional at the end: sigmoid
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

"""
class SegfaceMLP(nn.Module):
    
    Linear Embedding.
    Bringt alle Encoder Dimensionen auf 256. Brauche ich warscheinlich nicht
    

    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 256)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states
"""

class SegFace(nn.Module):
    def __init__(self, input_resolution):
        super(SegFaceCeleb, self).__init__()
        self.input_resolution = input_resolution
        self.model = model

        swin_v2 = swin_b(weights=None)
        # takes everything without the last head (classification header)
        self.backbone = torch.nn.Sequential(*(list(swin_v2.children())[:-1]))

        # load pretrained SegFace backbone weights in backbone
        ckpt = torch.load(".\pretrained_weights\SegFace_swin_celaba_512.pt", map_location="cpu")
        state_dict = ckpt["state_dict_backbone"]

        # remove "backbone"-prefix so that the keys match
        state_dict_stripped = {}
        for key, value in state_dict.items():
            if key.startswith("backbone."):
                new_key = key.replace("backbone.", "", 1)
                state_dict_stripped[new_key] = value
    
        # load weights into backbones
        missing, unexpected = self.backbone.load_state_dict(state_dict_stripped, strict=False)

        # log missing or unexpected keys
        if missing:   print("Missing keys:", len(missing))
        if unexpected: print("Unexpected keys:", len(unexpected))

        # targetlayer for exctracting feature maps
        self.target_layer_names = ['0.1', '0.3', '0.5', '0.7']
        self.multi_scale_features = []

        embed_dim = 1024
        out_chans = 256

        num_encoder_blocks = 4
        if self.model in ["swin_base", "swinv2_base", "convnext_base"]:
            hidden_sizes = [128, 256, 512, 1024] ### Swin Base and ConvNext Base
        
        # U-Net-like decoders require features from all four hierarchical levels.
        # → The hooks are used to tap into the layers directly inside.
        for name, module in self.backbone.named_modules():
            if name in self.target_layer_names:
                module.register_forward_hook(self.save_features_hook(name))
        
    # hook for extracting feature maps out of the target layers
    def save_features_hook(self, name):
        def hook(module, input, output):
            if self.model in ["swin_base", "swinv2_base", "swinv2_small", "swinv2_tiny"]:
                self.multi_scale_features.append(output.permute(0,3,1,2).contiguous()) ### Swin, Swinv2
        return hook

    def forward(self, x, labels, dataset):
        # clears the hook of the multi-scale feature map
        self.multi_scale_features.clear()
        
        _,_,h,w = x.shape
        features = self.backbone(x).squeeze()
        
        # takes the batch_size out of the las feature map
        batch_size = self.multi_scale_features[-1].shape[0]
        all_hidden_states = ()

        """
        for encoder_hidden_state, mlp in zip(self.multi_scale_features, self.linear_c):
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=self.multi_scale_features[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)
        
        fused_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1)) #### torch.Size([BS, 256, 128, 128])
        image_pe = self.pe_layer((fused_states.shape[2], fused_states.shape[3])).unsqueeze(0)
        """
        seg_output = self.MSUNetDecoder(fused_states)
    
        return seg_output

if __name__ == "__main__":
    input_resolution = 512
    model_name = "swin_base"
    model = SegFaceCeleb(input_resolution, model_name)
    
    batch_size = 4
    num_channels = 3
    height = 512
    width = 512

    x = torch.randn(batch_size, num_channels, height, width)
    
    labels = {
        "lnm_seg": torch.randn(batch_size, 5, 2)
    }
    
    dataset = torch.tensor([0,0,0,0])

    seg_output = model(x, labels, dataset)
    print("Segmentation Output Shape:", seg_output.shape)