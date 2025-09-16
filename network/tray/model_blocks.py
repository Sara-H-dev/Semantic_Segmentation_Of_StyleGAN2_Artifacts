import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.checkpoint as checkpoint
import numpy as np
import math

class MLP(nn.Module):
    r""" Multi-Layer Perceptron
    
    Args:
        input_dim (int): Number of input features
        hidden_dim (int): Number of hidden features
        out_dim (int): Number of output features
        act_layer (nn.GELu): Activation function
        drop (float): Dropout probability (0.0 means no dropout)
        dropout_on_output (bool): If True, apply dropout after the final layer
    """
    def __init__(
            self,
            input_dim: int, 
            hidden_dim: int | None = None,
            out_dim: int | None = None,
            act_layer: type[nn.Module] = nn.GELU, 
            drop: float = 0.0,
            dropout_on_output: bool = True,):
        
        super().__init__()
        out_dim = out_dim or input_dim
        hidden_dim = hidden_dim or input_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(drop)
        self.dropout_on_output = dropout_on_output

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        if self.dropout_on_output:
            x = self.drop(x)
        return x
    
class SwinTransBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        input_dim (int): Number of input channels (embedding dimension).
        input_resolution (tuple[int]): Spatial resolution of the input feature map (H, W)
        num_att_heads (int): Number of attention heads.
        window_size (int): Local window size for window-based multi-head self-attention. Default: 7
        shift_size (int): Shift size for shifted window mechanism (SW-MSA). Default: 0
        mlp_ratio (float): Ratio of hidden dimension to input dimension in MLP. Default: 4.0
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
            self, 
            input_dim: int,
            input_resolution: tuple[int, int], 
            num_att_heads: int, 
            window_size: int = 7, 
            shift_size: int = 0,
            mlp_ratio: float =4.0, 
            qkv_bias: bool = True, 
            qk_scale: float | None =None, 
            drop: float = 0.0, 
            attn_drop: float = 0.0, 
            drop_path: float = 0.0, 
            act_layer: type[nn.Module] = nn.GELU,
            norm_layer: type[nn.Module] = nn.LayerNorm, ):
        super().__init__()
        self.input_dim = input_dim
        self.input_resolution = input_resolution
        self.num_att_heads = num_att_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(input_dim)
        self.attn = WindowAttention(
            input_dim, window_size=to_2tuple(self.window_size), num_att_heads=num_att_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(input_dim)
        mlp_hidden_dim = int(input_dim * mlp_ratio)
        self.mlp = Mlp(in_features=input_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, input_resolution={self.input_resolution}, num_att_heads={self.num_att_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.input_dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.input_dim * self.input_dim * self.mlp_ratio
        # norm2
        flops += self.input_dim * H * W
        return flops