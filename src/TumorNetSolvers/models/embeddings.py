#Code adapted from: https://github.com/Aswathi-Varma/varivit  

from torch import nn
import torch


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
    def __init__(self, param_dim, embed_dim, experiment, hidden_dims=[64, 128, 256], N_patches=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.experiment = experiment
        self.param_dim = param_dim
        self.N_patches = N_patches
        self.extra_dims = 30  # For embed_concat mode

        method, mode = experiment

        def _build_mlp(input_dim, output_dim, hidden_dims):
            layers = []
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, output_dim))
            return nn.Sequential(*layers)

        if mode == "one_token":
            if method == "MLP":
                self.projection = _build_mlp(param_dim, embed_dim, hidden_dims)
            elif method == "Linear":
                self.projection = nn.Linear(param_dim, embed_dim)
            else:
                raise ValueError(f"Unsupported method: {method}")

        elif mode == "mul_token":
            #if method == "MLP":
            #    self.projection = _build_mlp(1, embed_dim, hidden_dims)
            self.projections = nn.ModuleList()
            if method == "MLP":
                for _ in range(param_dim):
                    self.projections.append(_build_mlp(1, embed_dim, hidden_dims))
            elif method == "Linear":
                for _ in range(param_dim):
                    self.projections.append(nn.Linear(1, embed_dim))
            elif method == "MLP_ext":
                self.projection = _build_mlp(param_dim, embed_dim * param_dim, hidden_dims)
            elif method == "Linear_ext":
                self.projection = nn.Linear(param_dim, embed_dim * param_dim)
            #elif method == "Linear":
                self.projection = nn.Linear(1, embed_dim)
            else:
                raise ValueError(f"Unsupported method for mul_token: {method}")

        elif mode == "embed_concat":
            out_dim = self.extra_dims
            if method == "MLP":
                self.projection = _build_mlp(param_dim, out_dim, hidden_dims)
            elif method == "Linear":
                self.projection = nn.Linear(param_dim, out_dim)
            else:
                raise ValueError(f"Unsupported method for embed_concat: {method}")

        elif mode == "embed_add":
            if N_patches is None:
                raise ValueError("N_patches must be specified for embed_add mode")
            out_dim = N_patches * embed_dim
            if method == "MLP":
                self.projection = _build_mlp(param_dim, out_dim, hidden_dims)
            elif method == "Linear":
                self.projection = nn.Linear(param_dim, out_dim)
            else:
                raise ValueError(f"Unsupported method for embed_add: {method}")

        else:
            raise ValueError(f"Unsupported mode: {mode}")


    def forward(self, x):
        method, mode = self.experiment
        B = x.size(0)

        if mode == "one_token":
            out = self.projection(x)             # (B, embed_dim)
            out = out.unsqueeze(1)               # (B, 1, embed_dim)
            return out

        elif mode == "mul_token":
            '''if method == "MLP" or method == "Linear":
                tokens = []
                for i in range(self.param_dim):
                    param_i = x[:, i:i+1]            # (B, 1)
                    token_i = self.projection(param_i)  # (B, embed_dim)
                    tokens.append(token_i.unsqueeze(1)) # (B, 1, embed_dim)
                out = torch.cat(tokens, dim=1)       # (B, param_dim, embed_dim)'''
            if method in {"MLP", "Linear"}:
                tokens = []
                for i in range(self.param_dim):
                    param_i = x[:, i:i+1]                     # (B, 1)
                    token_i = self.projections[i](param_i)    # (B, embed_dim)
                    tokens.append(token_i.unsqueeze(1))       # (B, 1, embed_dim)
                out = torch.cat(tokens, dim=1)                # (B, param_dim, embed_dim)
            elif method == "MLP_ext" or method == "Linear_ext":
                out = self.projection(x)           # (B, embed_dim * param_dim)
                out = out.view(B, self.param_dim, self.embed_dim)
            return out

        elif mode == "embed_concat":
            out = self.projection(x)             # (B, extra_dims)
            out = out.unsqueeze(1).expand(-1, self.N_patches, -1)  # (B, N_patches, extra_dims)
            return out

        elif mode == "embed_add":
            out = self.projection(x)             # (B, N_patches * embed_dim)
            out = out.view(B, self.N_patches, self.embed_dim)  # (B, N_patches, embed_dim)
            return out

        else:
            raise ValueError(f"Unsupported mode in forward: {mode}")