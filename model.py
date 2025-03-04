from functools import partial
from einops.einops import rearrange
from distutils.command.config import config
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
from torch.distributions.normal import Normal
import torch.nn.functional as nnf
import numpy as np
import configs_DualTM_Pyramid_crossattn as configs
import copy
import torch
from torch import nn
from mamba_ssm import Mamba
import torch.nn.functional as nnf
import pdb
import torch.nn.functional as F
try:
    from mamba import MambaLayer
except ModuleNotFoundError:
    pass
import matplotlib.pyplot as plt
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, L, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    B, H, W, L, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], L // window_size[2], window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows


def window_reverse(windows, window_size, H, W, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
        L (int): Length of image
    Returns:
        x: (B, H, W, L, C)
    """
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], L // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, L, -1)
    return x

class WindowAttention_CAM(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, rpe=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1 * 2*Wt-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_t = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_t]))  # 3, Wh, Ww, Wt
        coords_flatten = torch.flatten(coords, 1)  # 3, Wh*Ww*Wt
        self.rpe = rpe
        if self.rpe:
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wh*Ww*Wt, Wh*Ww*Wt
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww*Wt, Wh*Ww*Wt, 3
            relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww*Wt, Wh*Ww*Wt
            self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww, Wt*Ww) or None
        """
        B_, N, C = x.shape #(num_windows*B, Wh*Ww*Wt, C)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        #[1024, 4, 210, 24]
        k = k.view(2, -1, self.num_heads, N, C // self.num_heads) #[2, 512, 4, 210, 24]
        k = k[[1,0], :, :, :, :]
        k = k.view(-1, self.num_heads, N, C // self.num_heads) #[1024, 4, 210, 24]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) #[1024, 4, 210, 210]
        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1] * self.window_size[2],
                self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # Wh*Ww*Wt,Wh*Ww*Wt,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww*Wt, Wh*Ww*Wt
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            #[2, 512, 4, 210, 210] + [1, 512, 1, 210, 210]
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        v = v.view(2, -1, self.num_heads, N, C // self.num_heads) #[2, 512, 4, 210, 24]
        v = v[[1,0], :, :, :, :]
        v = v.view(-1, self.num_heads, N, C // self.num_heads) #[1024, 4, 210, 24]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C) # [1024, 210, 4, 24] --> [1024, 210, 96]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x




class  SwinTransformerBlock_CAM(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=(7, 7, 7), shift_size=(0, 0, 0),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, rpe=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= min(self.shift_size) < min(self.window_size), "shift_size must in 0-window_size, shift_sz: {}, win_size: {}".format(self.shift_size, self.window_size)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_CAM(
            dim, window_size=self.window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, rpe=rpe, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None
        self.T = None


    def forward(self, x, mask_matrix):
        H, W, T = self.H, self.W, self.T
        B, L, C = x.shape
        assert L == H * W * T, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, T, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_f = 0
        pad_r = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
        pad_b = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_h = (self.window_size[2] - T % self.window_size[2]) % self.window_size[2]
        x = nnf.pad(x, (0, 0, pad_f, pad_h, pad_t, pad_b, pad_l, pad_r))
        _, Hp, Wp, Tp, _ = x.shape

        # cyclic shift
        if min(self.shift_size) > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1], -self.shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2], C)  # nW*B, window_size*window_size*window_size, C

        # W-MCA/SW-MCA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], self.window_size[2], C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp, Tp)  # B H' W' L' C

        # reverse cyclic shift
        if min(self.shift_size) > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1], self.shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :T, :].contiguous()

        x = x.view(B, H * W * T, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x



class PatchMerging_MambaCat(nn.Module):
    r""" Patch Merging Layer.
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm, reduce_factor=2):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, (8//reduce_factor) * dim, bias=False)
        self.norm = norm_layer(8 * dim)


    def forward(self, x, H, W, T):
        """
        x: B, 2*H*W*T, C
        """
        B, L, C = x.shape
        assert L == 2 * H * W * T, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0 and T % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, 2, H, W, T, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1) or (T % 2 == 1)
        if pad_input:
            x = nnf.pad(x, (0, 0, 0, T % 2, 0, W % 2, 0, H % 2))

        x0 = x[:, :, 0::2, 0::2, 0::2, :]  # B 2 H/2 W/2 T/2 C
        x1 = x[:, :, 1::2, 0::2, 0::2, :]  # B 2 H/2 W/2 T/2 C
        x2 = x[:, :, 0::2, 1::2, 0::2, :]  # B 2 H/2 W/2 T/2 C
        x3 = x[:, :, 0::2, 0::2, 1::2, :]  # B 2 H/2 W/2 T/2 C
        x4 = x[:, :, 1::2, 1::2, 0::2, :]  # B 2 H/2 W/2 T/2 C
        x5 = x[:, :, 0::2, 1::2, 1::2, :]  # B 2 H/2 W/2 T/2 C
        x6 = x[:, :, 1::2, 0::2, 1::2, :]  # B 2 H/2 W/2 T/2 C
        x7 = x[:, :, 1::2, 1::2, 1::2, :]  # B 2 H/2 W/2 T/2 C
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)  # B H/2 W/2 T/2 8*C
        x = x.view(B, -1, 8 * C)  # B 2 * H/2*W/2*T/2 8*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.leaky_relu = nn.LeakyReLU(0.2)
     
        self.sn_conv1 = nn.utils.spectral_norm(nn.Conv3d(1, 32, kernel_size=3, stride=2))
        
        self.sn_conv2 = nn.utils.spectral_norm(nn.Conv3d(32, 64, kernel_size=3, stride=2))
        
        self.sn_conv3 = nn.utils.spectral_norm(nn.Conv3d(64, 128, kernel_size=3, stride=2))

        self.connected_layer1 = nn.Linear(in_features=12800, out_features=1024)
        self.connected_layer2 = nn.Linear(in_features=1024, out_features=64)
        self.connected_layer3 = nn.Linear(in_features=64, out_features=1)
     
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
      
        x = self.leaky_relu(self.sn_conv1(x.float()))
        x = self.leaky_relu(self.sn_conv2(x))
        x = self.leaky_relu(self.sn_conv3(x))

        x_batch, x_c, h, w,d = x.size()
    
        x = x.contiguous().view(x_batch, -1)
        x = self.connected_layer1(x)
        x = self.connected_layer2(x)
        x = self.connected_layer3(x)

        return self.sigmoid(x)
    
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 depth_cam,
                 num_heads_cam,
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 window_size_cam=(7, 7, 7),
                 mlp_ratio_cam=4.,
                 qkv_bias_cam=True,
                 qk_scale=None,
                 rpe=True,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 drop_path_cam=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 if_cam = True,
                 if_mix_mamba = False,
                 if_selfskip = False,
                 if_crossskip = False,
                 if_crossaddselfskip = True, 
                 d_state=16, 
                 d_conv=4, 
                 expand=2,):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.pat_merg_rf = pat_merg_rf
        self.if_cam = if_cam
        self.if_mix_mamba = if_mix_mamba
        self.if_selfskip = if_selfskip
        self.if_crossskip = if_crossskip
        self.if_crossaddselfskip = if_crossaddselfskip
        # build blocks
        
        if self.if_cam:
            self.blocks_cam = nn.ModuleList([
                SwinTransformerBlock_CAM(
                    dim=dim,
                    num_heads=num_heads_cam,
                    window_size=window_size_cam,
                    shift_size=(0, 0, 0) if (i % 2 == 0) else (window_size_cam[0] // 2, window_size_cam[1] // 2, window_size_cam[2] // 2),
                    mlp_ratio=mlp_ratio_cam,
                    qkv_bias=qkv_bias_cam,
                    qk_scale=qk_scale,
                    rpe=rpe,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path_cam[i] if isinstance(drop_path_cam, list) else drop_path_cam,
                    norm_layer=norm_layer,)
                for i in range(depth_cam)])

        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.mamba2 = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state * 2,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand * 2,  # Block expansion factor
        )
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, reduce_factor=self.pat_merg_rf)
        else:
            self.downsample = None

    def forward(self, x, H, W, T):
        """ Forward function.
        Args:
            x: Input feature, tensor size (B, H*W*T, C).
            H, W, T: Spatial resolution of the input feature.
        """

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size[0])) * self.window_size[0]
        Wp = int(np.ceil(W / self.window_size[1])) * self.window_size[1]
        Tp = int(np.ceil(T / self.window_size[2])) * self.window_size[2]
        img_mask = torch.zeros((1, Hp, Wp, Tp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        t_slices = (slice(0, -self.window_size[2]),
                    slice(-self.window_size[2], -self.shift_size[2]),
                    slice(-self.shift_size[2], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                for t in t_slices:
                    img_mask[:, h, w, t, :] = cnt
                    cnt += 1
        # [512, 5, 6, 7, 1]
        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        #attn_mask [512, 210, 210]
        ## self_attention_block, for a depth

        B, C = x.shape[0], x.shape[-1]
        assert C == self.dim
         
        x_norm = self.norm(x)
        if x_norm.dtype == torch.float16:
            x_norm = x_norm.type(torch.float32)
        x_mamba = self.mamba(x_norm)
        x = x_mamba.type(x.dtype)
        ## cross_attention_block, for a batchsize 2
        fea_self = x

        if self.if_cam:
            # print('you have cross attention module!!!!!')
            for blk in self.blocks_cam:
                blk.H, blk.W, blk.T = H, W, T
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x, attn_mask)
                else:
                    x = blk(x, attn_mask)
            fea_fordown = fea_self + x
        else:
            fea_fordown = fea_self
        
        if self.if_mix_mamba:
            # print('you have cross attention module!!!!!')
            x_norm = self.norm(fea_self)
            if x_norm.dtype == torch.float16:
                x_norm = x_norm.type(torch.float32)
            x_mamba2 = self.mamba2(x_norm)
            x = x_mamba2.type(fea_self.dtype)
            ## cross_attention_block, for a batchsize 2
            fea_self_2 = x
            fea_fordown = fea_self + fea_self_2
            print('resmamba')
        else:
            fea_fordown = fea_self

        if self.downsample is not None:
            x_down = self.downsample(fea_fordown, H, W, T)
            Wh, Ww, Wt = (H + 1) // 2, (W + 1) // 2, (T + 1) // 2
            if self.if_selfskip:
                return fea_self, H, W, T, x_down, Wh, Ww, Wt
            if self.if_crossskip:
                return x, H, W, T, x_down, Wh, Ww, Wt
            if self.if_crossaddselfskip:
                return fea_fordown, H, W, T, x_down, Wh, Ww, Wt
        else:
            if self.if_selfskip:
                return fea_self, H, W, T, fea_self, H, W, T
            if self.if_crossskip:
                return x, H, W, T, x, H, W, T
            if self.if_crossaddselfskip:
                return fea_fordown, H, W, T, fea_fordown, H, W, T


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_3tuple(patch_size) #(4,4,4)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W, T = x.size()
        if T % self.patch_size[2] != 0:
            x = nnf.pad(x, (0, self.patch_size[2] - T % self.patch_size[2]))
        if W % self.patch_size[1] != 0:
            x = nnf.pad(x, (0, 0, 0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = nnf.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww Wt (2, 96, 40, 48, 56)
        if self.norm is not None:
            Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2) #(2,107520,96)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww, Wt)

        return x

class SinusoidalPositionEmbedding(nn.Module):
    '''
    Rotary Position Embedding
    '''
    def __init__(self,):
        super(SinusoidalPositionEmbedding, self).__init__()

    def forward(self, x):
        batch_sz, n_patches, hidden = x.shape
        position_ids = torch.arange(0, n_patches).float().cuda()
        indices = torch.arange(0, hidden//2).float().cuda()
        indices = torch.pow(10000.0, -2 * indices / hidden)
        embeddings = torch.einsum('b,d->bd', position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (1, n_patches, hidden))
        return embeddings

class SinPositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(SinPositionalEncoding3D, self).__init__()
        channels = int(np.ceil(channels/6)*2)
        if channels % 2:
            channels += 1
        self.channels = channels
        self.inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        #self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        tensor = tensor.permute(0, 2, 3, 4, 1)
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x,y,z,self.channels*3),device=tensor.device).type(tensor.type())
        emb[:,:,:,:self.channels] = emb_x
        emb[:,:,:,self.channels:2*self.channels] = emb_y
        emb[:,:,:,2*self.channels:] = emb_z
        emb = emb[None,:,:,:,:orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return emb.permute(0, 4, 1, 2, 3)

class SwinTransformer_CAM(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (tuple): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=(7, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 depths_cam=[2, 2, 6, 2],
                 num_heads_cam=[3, 6, 12, 24],
                 window_size_cam=(7, 7, 7),
                 mlp_ratio_cam=4.,
                 qkv_bias_cam=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 spe=False,
                 rpe=True,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False,
                 pat_merg_rf=2,
                 if_cam = True,
                 if_mix_mamba = False,
                 if_selfskip = True,
                 if_crossskip = True,
                 if_crossaddselfskip = True,
                 d_state=16,
                 d_conv=4,
                 expand=2,):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.spe = spe
        self.rpe = rpe
        self.if_cam = if_cam
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
       
        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_3tuple(self.pretrain_img_size)
            patch_size = to_3tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1], pretrain_img_size[2] // patch_size[2]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1], patches_resolution[2]))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        elif self.spe:
            self.pos_embd = SinPositionalEncoding3D(embed_dim).cuda()
            #self.pos_embd = SinusoidalPositionEmbedding().cuda()
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        if self.if_cam:
            dpr_cam = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_cam))]  # stochastic depth decay rule
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                depth_cam=depths_cam[i_layer],
                                num_heads_cam=num_heads_cam[i_layer],
                                window_size_cam=window_size_cam,
                                mlp_ratio_cam=mlp_ratio_cam,
                                qkv_bias_cam=qkv_bias_cam,
                                rpe = rpe,
                                qk_scale=qk_scale,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                drop_path_cam=dpr_cam[sum(depths_cam[:i_layer]):sum(depths_cam[:i_layer + 1])] if self.if_cam else None,
                                norm_layer=norm_layer,
                                downsample=PatchMerging_MambaCat if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint,
                                pat_merg_rf=pat_merg_rf,
                                if_cam = if_cam,
                                if_mix_mamba = if_mix_mamba,
                                if_selfskip = if_selfskip,
                                if_crossskip = if_crossskip,
                                if_crossaddselfskip = if_crossaddselfskip, 
                                d_state=d_state,
                                d_conv=d_conv,
                                expand=expand,)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        #[C, 2C, 4C, 8C]
        self.num_features = num_features

        # add a norm layer for each output
        
        for i_layer in out_indices: #(0,1,2,3)
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x) # (2,96,40,48,56)
        Wh, Ww, Wt = x.size(2), x.size(3), x.size(4)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = nnf.interpolate(self.absolute_pos_embed, size=(Wh, Ww, Wt), mode='trilinear')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww*Wt C
        elif self.spe:
            x = (x + self.pos_embd(x)).flatten(2).transpose(1, 2)
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x) # (2, 107520, 96)

        ## (2, 107520, 96)
        moving_source = x[0:1, :, :] # (1,:,C)
        fixed_source = x[1:2, :, :] # (1,:,C)
        x_F = torch.cat((fixed_source,moving_source), dim=1)  
        x_M = torch.cat((moving_source,fixed_source), dim=1) 

        # x_F = torch.cat((fixed_source,fixed_source), dim=1)  
        # x_M = torch.cat((moving_source,moving_source), dim=1)         
        ##
        x = torch.cat((x_M,x_F), dim=0) 

        outs = []
        for i in range(self.num_layers): #(0,1,2,3)
            layer = self.layers[i] # Transformer block + patch merging
            # x_out is features before patch_merging
            x_out, H, W, T, x, Wh, Ww, Wt = layer(x, Wh, Ww, Wt)
            ## add a convolutional layer after patchmerging.
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)
                out = x_out.view(-1, 2, H, W, T, self.num_features[i]).permute(0, 1, 5, 2, 3, 4).contiguous()
                out = out.view(-1, 2*self.num_features[i], H, W, T)
                outs.append(out)
        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer_CAM, self).train(mode)
        self._freeze_stages()















class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderBlock_DTP(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels * 2,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            skipmoving = skip[0].unsqueeze(0)
            skipfixed = skip[1].unsqueeze(0)
            x = torch.cat([skipmoving, x, skipfixed], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class JConv(nn.Module):
    """Joint Convolutional blocks

    Args:
        'x' (torch.Tensor): (N, C, H, W)
    """

    def __init__(self, in_channels, out_channels,stride=1):
        super(JConv, self).__init__()
        self.feat_trans = ConvBlock(in_channels, out_channels, stride=stride)
        self.dwconv = DWConv(out_channels)
        self.mlp = MLP(out_channels, bias=True)

    def forward(self, x):
        x = self.feat_trans(x)
        x = x + self.dwconv(x)
        x = x + self.mlp(x)
        return x

class MLP(nn.Module):
    """MLP Layer

    Args:
        'x': (torch.Tensor): (N, C, H, W)
    """
    def __init__(self, out_channels, bias=True):
        super().__init__()
        self.conv1 = nn.Conv3d(out_channels, out_channels * 2, kernel_size=1, bias=bias)
        self.act = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv3d(out_channels * 2, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, 'Conv3d')
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out
    
class DWConv(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, stride=1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, in_channels, 3, stride=1,padding=1, groups=in_channels)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class RegistrationHead_2conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d_1 = Conv3dReLU(in_channels, int(in_channels/2), kernel_size=kernel_size, padding=kernel_size // 2) 
        conv3d = nn.Conv3d(int(in_channels/2), out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d_1, conv3d)
        # super().__init__(conv3d)

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.

    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)



    
class StructureAttention(nn.Module):
    # This class is implemented by [LoFTR](https://github.com/zju3dv/LoFTR).
    def __init__(self, d_model, nhead):
        super(StructureAttention, self).__init__()
        self.dim = d_model // nhead
        self.nhead = nhead
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message
    
def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1

class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        Q = self.feature_map(queries)
        K = self.feature_map(keys)
        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()

class CrossmodalSemanticsCalibration(nn.Module):
    """Cross-modal semantic calibration

    Args:
        'feat' (torch.Tensor): (N, L)
        'attention' (torch.Tensor): (N, L, C)
    """

    def __init__(self, dim, mlp_ratio=2, qkv_bias=False):
        super(CrossmodalSemanticsCalibration, self).__init__()
        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj_out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim*2)
        self.mlp = Mlp(dim*2, dim*2,dim)

        self.cross1=StructureAttention(96,8)
        # self.cross2=StructureAttention(96,8)

    def forward(self, mr_seg_feat_flatten, ct2mr_attention_flatten,warp_ctfeat):

        # shortcut = mr_seg_feat_flatten
        mr_seg_feat_flatten = self.norm1(mr_seg_feat_flatten)
        shortcut = self.qkv(mr_seg_feat_flatten)
        mr_seg_feat_flatten = self.qkv2(shortcut)
        

        x=self.cross1(mr_seg_feat_flatten,warp_ctfeat)
        x=torch.cat([x,shortcut],dim=2)
        # bag=self.cross2(warp_ctfeat,mr_seg_feat_flatten)*(1-ct2mr_attention_flatten)


        # x = torch.einsum('nlc, nls -> nls', ct2mr_attention_flatten, mr_seg_feat_flatten)
        # x = self.proj_out(sem)
        # x = x + shortcut
        x =  self.mlp(self.norm2(x))
        # shortcut = feat
        # feat = self.norm1(feat)
        # feat = self.qkv(feat)
        # x = torch.einsum('nl, nlc -> nlc', attention, feat)
        # x = self.proj_out(x)
        # x = x + shortcut
        # x = x + self.mlp(self.norm2(x))
        return x
def convert_to_coord_format3d(b, h, w, d, device='cpu'):
        x_channel = torch.linspace(-1, 1, w, device=device).view(1, 1, 1, -1, 1).repeat(b, 1, h, 1,
                                                                                        d)  # torch.Size([1, 1, 24, 24, 24])
        y_channel = torch.linspace(-1, 1, h, device=device).view(1, 1, -1, 1, 1).repeat(b, 1, 1, w, d)
        z_channel = torch.linspace(-1, 1, d, device=device).view(1, 1, 1, 1, -1).repeat(b, 1, h, w,
                                                                                        1)  # torch.Size([1, 1, 48, 48, 48])
        return torch.cat((x_channel, y_channel, z_channel), dim=1)

class Positional(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))
    
class PatchEmbedmatch(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, with_pos=True):
        super().__init__()
        img_size = img_size
        patch_size = [patch_size,patch_size,patch_size]
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W,self.D = img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]
        self.num_patches = self.H * self.W * self.D
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2,  patch_size[2] // 2))

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = Positional(embed_dim)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        _, _, H, W,D  = x.shape       
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        return x, H, W,D

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, cross = False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, cross= cross)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp2(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W,D):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W,D))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W,D))

        return x
class DWConv2(nn.Module):
    def __init__(self, dim=768):
        super(DWConv2, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W,D):
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, H, W,D)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
class Mlp2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv2(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x,H,W,D):
        x = self.fc1(x)
        x = self.dwconv(x, H, W,D)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, cross= False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.cross = cross

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W,D):
        B, N, C = x.shape
        if self.cross == True:
            MiniB = B // 2
            #cross attention
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q1,q2 = q.split(MiniB)

            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W,D)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            
            k1, k2 = kv[0].split(MiniB)
            v1, v2 = kv[1].split(MiniB)

            attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)

            attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)

            x1 = (attn1 @ v2).transpose(1, 2).reshape(MiniB, N, C)
            x2 = (attn2 @ v1).transpose(1, 2).reshape(MiniB, N, C)

            x = torch.cat([x1, x2], dim=0)
        else:
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W,D)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)

          

            attn = self.attn_drop(attn)

            

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
class AttentionBlock(nn.Module):
    def __init__(self, img_size=224, in_chans=1, embed_dims=128, patch_size=7, num_heads=1, mlp_ratios=4, sr_ratios=8,
                 qkv_bias=True, drop_rate=0., norm_layer=partial(nn.LayerNorm, eps=1e-6),stride=2, depths=1, cross=[False,False,True]):
        super().__init__()
        self.patch_embed = PatchEmbedmatch(img_size=img_size, patch_size=patch_size, stride = stride, in_chans=in_chans,
                                              embed_dim=embed_dims)
        self.block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,sr_ratio= sr_ratios,
            drop=drop_rate, drop_path=0, norm_layer=norm_layer, cross=cross[i])
            for i in range(depths)])
        self.norm = norm_layer(embed_dims)

    def forward(self, x):
        B = x.shape[0]
        x, H, W,D  = self.patch_embed(x)
        for i, blk in enumerate(self.block):
            x = blk(x,H,W,D)
        x = self.norm(x)
        x = x.reshape(B, H, W,D, -1).permute(0, 4, 1, 2,3).contiguous()

        return x
    
class MultimodalSemanticsEncoding(nn.Module):
    """Cross-modal semantic calibration

    Args:
        'feat' (torch.Tensor): (N, L)
        'attention' (torch.Tensor): (N, L, C)
    """

    def __init__(self,dim,feat_size):
        super(MultimodalSemanticsEncoding, self).__init__()
        self.feat_size=feat_size
        self.grid_encoder=nn.Sequential(JConv(3,dim))
        self.semantic_embedding=JConv(dim,dim)
        self.feat_grid_attention0=AttentionBlock(img_size=[64,40,48], patch_size=7, num_heads= 4, mlp_ratios=4,  in_chans=128,
                                               embed_dims=128,stride=2,sr_ratios=2,depths=1, cross=[False])
     
    def forward(self, feat):
        grid =convert_to_coord_format3d(1,self.feat_size[0],self.feat_size[1],self.feat_size[2],
                                              torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        grid=self.grid_encoder(grid)

        semantic_grid = grid * feat
        semantic_grid = self.semantic_embedding(semantic_grid)
        semantic_grid=self.feat_grid_attention0(semantic_grid)
      
        return semantic_grid

class SLENet(nn.Module):
    def __init__(self, config):
        '''
        SLENet Model
        '''
        super(SLENet, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        # if_selfskip = config.if_selfskip
        # self.if_selfskip = if_selfskip
        # if_crossskip = config.if_crossskip
        # self.if_crossskip = if_crossskip
        # if_crossaddselfskip = config.if_crossaddselfskip
        # self.if_crossaddselfskip = if_crossaddselfskip
        embed_dim = config.embed_dim
        self.sup_level = config.sup_level
        self.num_layers = len(config.depths)
        self.img_size = [192, 160, 192]
        self.is_diff = config.is_diff
        # self.is_val = is_val
        self.transformer = SwinTransformer_CAM(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           depths_cam=config.depths_cam,
                                           num_heads_cam=config.num_heads_cam,
                                           window_size_cam=config.window_size_cam,
                                           mlp_ratio_cam=config.mlp_ratio_cam,
                                           qkv_bias_cam=config.qkv_bias_cam,
                                           ape=config.ape,
                                           spe=config.spe,
                                           rpe=config.rpe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           if_cam = config.if_crossam,
                                           if_mix_mamba = config.if_mix_mamba,
                                           if_selfskip = config.if_selfskip,
                                           if_crossskip = config.if_crossskip,
                                           if_crossaddselfskip = config.if_crossaddselfskip
                                           )
        
        self.bottle_conv = Conv3dReLU(embed_dim*16*2, embed_dim*8, 3, 1, use_batchnorm=False)
        self.up0 = DecoderBlock_DTP(embed_dim*8, embed_dim*4, skip_channels=embed_dim*4*2 if if_transskip else 0, use_batchnorm=False)
        self.up1 = DecoderBlock_DTP(embed_dim*4, embed_dim*2, skip_channels=embed_dim*2*2 if if_transskip else 0, use_batchnorm=False)  # 384, 20, 20, 64
        self.up2 = DecoderBlock_DTP(embed_dim*2, embed_dim, skip_channels=embed_dim*2 if if_transskip else 0, use_batchnorm=False)
        self.up3 = DecoderBlock_DTP(embed_dim, embed_dim//2, skip_channels= embed_dim // 2 if if_convskip else 0, use_batchnorm=False)  # 384, 80, 80, 128
        self.up4 = DecoderBlock_DTP(embed_dim//2, config.reg_head_chan, skip_channels=config.reg_head_chan if if_convskip else 0, use_batchnorm=False)  # 384, 160, 160, 256
        self.c1 = Conv3dReLU(1, embed_dim//2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(1, config.reg_head_chan, 3, 1, use_batchnorm=False)
        de_sup_num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        de_sup_num_features.reverse()
        de_sup_num_features = de_sup_num_features + list([embed_dim//2]) + list([config.reg_head_chan])
        self.de_sup_num_features = de_sup_num_features # [768, 384, 192, 96, 48, 16]
   
        # add convolutional layer for each feature output --registration supervision
        for i_reg_layer in self.sup_level:
            sup_reg_head = RegistrationHead_2conv(
                            in_channels=self.de_sup_num_features[i_reg_layer],
                            out_channels=3,
                            kernel_size=3,)
            sup_reg_head_name = f'reg{i_reg_layer}'
            self.add_module(sup_reg_head_name, sup_reg_head)
            sup_ST_layer = SpatialTransformer([int(i/(2 ** (5 - i_reg_layer))) for i in self.img_size])
            sup_ST_layer_name = f'regST{i_reg_layer}'
            self.add_module(sup_ST_layer_name, sup_ST_layer)
            if self.is_diff:
                sup_SS_layer = VecInt([int(i/(2 ** (5 - i_reg_layer))) for i in self.img_size], config.int_steps)
                sup_SS_layer_name = f'regSS{i_reg_layer}'
                self.add_module(sup_SS_layer_name, sup_SS_layer)
        
        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.up_tri = nn.Upsample(scale_factor=2, mode="trilinear")

        self.spatial_trans_down1 = SpatialTransformer((48, 40, 48),mode='nearest')
        # self.spatial_trans_down = SpatialTransformer((8, 24, 28),mode='nearest')

        self.seg_up0 = DecoderBlock(embed_dim*16, embed_dim*4, skip_channels=embed_dim*8 if if_transskip else 0, use_batchnorm=False)
        self.seg_up1 = DecoderBlock(embed_dim*4, embed_dim*2, skip_channels=embed_dim*4 if if_transskip else 0, use_batchnorm=False)  # 384, 20, 20, 64
        self.seg_up2 = DecoderBlock(embed_dim*2, embed_dim, skip_channels=embed_dim*2 if if_transskip else 0, use_batchnorm=False)
        self.seg_head_ct = nn.Sequential(JConv(96,24),JConv(24,24),JConv(24,1),nn.Sigmoid())
        self.seg_head_mr = nn.Sequential(JConv(96,24),JConv(24,24),JConv(24,1),nn.Sigmoid())
      

        self.csc=CrossmodalSemanticsCalibration(96)

        self.upp=Conv3dReLU(128*2, 16, 3, 1, use_batchnorm=False)
        self.upp2=Conv3dReLU(16, 16, 3, 1, use_batchnorm=False)
        # self.up=Conv3dReLU(384*2, 192, 3, 1, use_batchnorm=False)
        # self.up=nn.Sequential(JConv(384*2,384),JConv(384,192))

        self.mse=MultimodalSemanticsEncoding(128,(48, 40, 48))
        self.proj=Conv3dReLU(32, 16, 3, 1, use_batchnorm=False)
        # self.eeu=EEU(16)

        # self.corr=CCorrM()

        # self.csc_selfattention=StructureAttention(384,8)

        self.pos_embd = SinPositionalEncoding3D(192).cuda()

     

        # self.cscencoder=nn.Sequential(JConv(1,192),JConv(192,384))
        # self.ktm=KTM()
        # self.tanh=nn.Tanh()
        
    def forward(self, x, inmoving_mask,infixed_mask,is_eval=False):
        de_out_fea = []
      
        moving_source = x[0:1, :, :, :, :] # (1,1,160,192,224)
        fixed_source = x[1:2, :, :, :, :] # (1,1,160,192,224)
      
        if self.if_convskip:
            x_s5 = x.clone()# (2,1,160,192,224)
            x_s4 = self.avg_pool(x) #(2, 1, 80, 96, 112)
            f4 = self.c1(x_s4) #(2, 48, 80, 96, 112)
            f5 = self.c2(x_s5) #(2, 16, 160, 192, 224)
        else:
            f4 = None
            f5 = None
        out_feats = self.transformer(x)

        if self.if_transskip:
            f1 = out_feats[-2] #(2,384,10,12,14) features before patch merging and after norm layer
            f2 = out_feats[-3] #(2,192,20,24,28)
            f3 = out_feats[-4] #(2,96, 40,48,56)

        else:
            f1 = None
            f2 = None
            f3 = None

        batch_out_embed = out_feats[-1] #(2,768,5,6,7)
        movingout_embedding = batch_out_embed[0].unsqueeze(0) #(1,768,5,6,7)
        fixedout_embedding = batch_out_embed[1].unsqueeze(0)  #(1,768,5,6,7)
        catout_embedding = torch.cat([movingout_embedding, fixedout_embedding], dim=1) #(1, 1536, 5,6,7)
        x = self.bottle_conv(catout_embedding) #(1,768,5,6,7)
        de_out_fea.append(x)
        x = self.up0(x, f1) # (1,384,10,12,14)
        de_out_fea.append(x) 
        x = self.up1(x, f2)
        de_out_fea.append(x) #(1,192,20,24,28)
        x = self.up2(x, f3)
        de_out_fea.append(x) #(1,96,40,48,56)
        x = self.up3(x, f4)
        de_out_fea.append(x) #(1,48,80,96,112)
        x = self.up4(x, f5)
        de_out_fea.append(x) #(1,16,160,192,224)
        warp_moving = []
        fix_img = []
        out_velocity = []
        out_disp = []
        velocity_history = [0]
        disp_history = [0]


      
     
        if self.is_diff:
            for j, i_reg_layer in enumerate(self.sup_level):
                # conv + ss + ST    # up_tri            
                output_cov_layer = getattr(self, f'reg{i_reg_layer}')
                output_SS_layer = getattr(self, f'regSS{i_reg_layer}')
                output_ST_layer = getattr(self, f'regST{i_reg_layer}')
                compose_v = output_cov_layer(de_out_fea[i_reg_layer])
                compose_v = compose_v + velocity_history[j]
                if i_reg_layer < self.sup_level[-1]:
                    fac = 2 ** (int(self.sup_level[j + 1] - self.sup_level[j]))
                    # self.up_tri = nn.Upsample(scale_factor=fac, mode="trilinear")
                    output_v_up = self.up_tri(compose_v) * fac
                    velocity_history.append(output_v_up)
                if is_eval:
                    if i_reg_layer == self.sup_level[-1]:
                        output_disp = output_SS_layer(compose_v)
                        out = output_ST_layer(moving_source, output_disp)
                        warp_moving.append(out)
                        fix_img.append(fixed_source)
                        out_velocity.append(compose_v)
                        out_disp.append(output_disp) 
                        # print('For Testing!!!!!')
                else:
                    y_down = nnf.interpolate(fixed_source, scale_factor= 1/2**(5 - i_reg_layer), mode='trilinear')
                    output_disp = output_SS_layer(compose_v)
                    out = output_ST_layer(moving_source, output_disp)
                    warp_moving.append(out)
                    fix_img.append(y_down)
                    out_velocity.append(compose_v)
                    out_disp.append(output_disp)
        else:
            a=1


        x_seg = self.seg_up0(batch_out_embed, f1)
        x_seg = self.seg_up1(x_seg, f2)
        x_seg = self.seg_up2(x_seg, f3)
        mr_seg_feat, ct_seg_feat = x_seg.split(1)
        ct_seg = self.seg_head_ct(ct_seg_feat)
        raw_mrfeat=mr_seg_feat
       

       
        down_flow = F.interpolate(-out_velocity[0]/4, size=ct_seg.size()[2:], mode='trilinear')
        ct2mr_attention = self.spatial_trans_down1(ct_seg,down_flow) 
        warp_ctfeat = self.spatial_trans_down1(ct_seg_feat,down_flow)


    

        h = mr_seg_feat.shape[2]
        w = mr_seg_feat.shape[3]
        d = mr_seg_feat.shape[4]

        mr_seg_feat_flatten = rearrange(mr_seg_feat+self.pos_embd(mr_seg_feat), 'n c h w d-> n (h w d) c') #[1,15360,96]
        # ct2mr_attention=self.cscencoder(ct2mr_attention)
        ct2mr_attention_flatten = rearrange(ct2mr_attention, 'n c h w d-> n (h w d) c')
        warp_ctfeat = rearrange(warp_ctfeat+self.pos_embd(warp_ctfeat), 'n c h w d-> n (h w d) c') #[3,15360,1] 
        mr_feat=self.csc(mr_seg_feat_flatten,ct2mr_attention_flatten,warp_ctfeat)
        mr_feat = rearrange(mr_feat, 'n (h w d) c -> n c h w d', h=h, w=w, d=d)
     
        


      
        mr_raw = self.seg_head_ct(raw_mrfeat)
        mr_seg = self.seg_head_mr(mr_seg_feat)

        mr_ssr_feat=self.mse(mr_seg)
        ct_ssr_feat=self.mse(ct_seg)

        

        



        ssr_feat=self.upp(torch.cat((ct_ssr_feat,mr_ssr_feat),dim=1))
        ssr_feat=F.interpolate(ssr_feat, size=x.size()[2:], mode='trilinear')
        ssr_feat=self.upp2(ssr_feat)
        x=self.proj(torch.cat((x,ssr_feat),dim=1))


        out_velocity=output_cov_layer(x)
        output_disp = output_SS_layer(out_velocity)
        warp_moving = output_ST_layer(moving_source, output_disp)
        
                   

        return [warp_moving], fix_img, [out_velocity], [output_disp], ct_seg, mr_raw, mr_feat, ct_seg_feat,mr_raw


CONFIGS = {
    'Dual_TransMorph_Mamba': configs.get_3DDual_TransMorph_Mamba_config(),
    'SLENet': configs.get_SLENet_config(),
    'TransMorph-No-Conv-Skip': configs.get_3DTransMorphNoConvSkip_config(),
    'TransMorph-No-Trans-Skip': configs.get_3DTransMorphNoTransSkip_config(),
    'TransMorph-No-Skip': configs.get_3DTransMorphNoSkip_config(),
    'TransMorph-Lrn': configs.get_3DTransMorphLrn_config(),
    'TransMorph-Sin': configs.get_3DTransMorphSin_config(),
    'TransMorph-No-RelPosEmbed': configs.get_3DTransMorphNoRelativePosEmbd_config(),
    'TransMorph-Large': configs.get_3DTransMorphLarge_config(),
    'TransMorph-Small': configs.get_3DTransMorphSmall_config(),
    'TransMorph-Tiny': configs.get_3DTransMorphTiny_config(),
}
