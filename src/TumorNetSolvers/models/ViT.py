#inspired from https://github.com/chinmay5/vit_ae_plus_plus and https://github.com/MLI-lab/transformers_for_imaging
#with major modfifications

import torch
from torch import nn
from functools import partial
from collections import OrderedDict
from TumorNetSolvers.models.positional_encoding import get_3d_sincos_pos_embed
from TumorNetSolvers.models.embeddings import ParameterEmbedding, traid, PatchEmbed3D


class Attention(nn.Module):
    """
    Multi-head attention mechanism.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        attn_weights = attn.clone()

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_weights


class Mlp3D(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
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


class Block(nn.Module):
    """
    Transformer block with attention and MLP.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp3D(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x



class CombinedVisionTransformer3D(nn.Module):
    """
    Combined Vision Transformer for 3D data with embedding of both image patches and parameter tokens.
    """
    def __init__(self, max_volume_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=384, depth=12,
                num_heads=6, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed3D, norm_layer=None,
                act_layer=None, weight_init='', global_pool=False, param_dim=10):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 3 if distilled else 2  # Account for param_embed token
        self.patch_size = traid(patch_size)
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.max_volume_size = traid(max_volume_size)

        self.patch_embed = embed_layer(max_volume_size=max_volume_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.param_embed = ParameterEmbedding(param_dim, embed_dim)

        max_num_patches = self.patch_embed.max_num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, max_num_patches + self.num_tokens, embed_dim), requires_grad=False)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)] 
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, 
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], 
                norm_layer=norm_layer, act_layer=act_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        output_shape = in_chans * (max_volume_size // self.patch_size[0]) *self.patch_size[0]* (max_volume_size // self.patch_size[1])*self.patch_size[1] * (max_volume_size // self.patch_size[2])*self.patch_size[2]

        self.head = nn.Linear(self.num_features, output_shape)

        self.initialize_weights()

    def initialize_weights(self):
        max_num_patches = self.patch_embed.max_num_patches
        pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], round(max_num_patches ** (1 / 3)), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x, param_token):
        x = self.patch_embed(x)  
        B, Num_Patches, _ = x.shape 
        param_tokens = self.param_embed(param_token) 
        x = torch.cat((x, param_tokens), dim=1)
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, x, param_token):
        B, C, D, H, W = x.shape
        x = self.forward_features(x, param_token)

        # Use the head to map back to the original shape
        x = self.head(x[:, 0])  # Use only CLS token for head
        x = x.view(B, C, D, H, W)  # Reshape to match input dimensions
        return x

