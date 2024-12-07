#Code adapted from: https://github.com/Aswathi-Varma/varivit  

from torch import nn



def traid(t):
    return t if isinstance(t, tuple) else (t, t, t)

class PatchEmbed3D(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, max_volume_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        self.max_volume_size = traid(max_volume_size)

        self.patch_size = traid(patch_size)

        self.max_grid_size = (self.max_volume_size[0] // self.patch_size[0], self.max_volume_size[1] // self.patch_size[1], self.max_volume_size[2] // self.patch_size[2])
        #ie nbr of patches per dim (can be different along each dim)
        self.max_num_patches =  self.max_grid_size[0] * self.max_grid_size[1] * self.max_grid_size[2]
        self.flatten = flatten #param function taking bool for optional flattening

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()  #with optional layer normalization

    def forward(self, x):
        B, C, L, H, W = x.shape   #batch size, chs, length, height, width
        x = self.proj(x) #ie conv3D
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC  #flatten here flattens starting from 2 ie convert H,W to one dimBNC isperfect/expected input shape by Transformers: Batch_size, nbr of patches, and chs
        x = self.norm(x)
        return x

class ParameterEmbedding(nn.Module):
    def __init__(self, param_dim, embed_dim):
        super().__init__()
        self.projection = nn.Linear(param_dim, embed_dim)

    def forward(self, x):
        param_embedding = self.projection(x).unsqueeze(1)  
        return param_embedding