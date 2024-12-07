# adapted from: https://github.com/chinmay5/vit_ae_plus_plus/tree/main

import numpy as np

def get_3d_sincos_pos_embed(embed_dim, grid_size, num_param_tokens=1, cls_token=False):
    """
    Generate 3D sine-cosine positional embeddings for a 3D grid.

    Args:
        embed_dim (int): Dimensionality of the embedding vector for each position.
        grid_size (int): Size of the grid in each dimension (assumes cubic grid).
        num_param_tokens (int): Number of parameter tokens to prepend. Default is 1.
        cls_token (bool): Whether to include a class token at the beginning of the embeddings. Default is False.

    Returns:
        numpy.ndarray: Positional embeddings of shape [num_tokens, embed_dim], where `num_tokens` includes 
                       parameter tokens, grid positions, and optionally a class token.
    """
    # Generate 3D grid
    grid_l = np.arange(grid_size, dtype=np.float32)
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_l, grid_h, grid_w)  # Create a 3D grid
    grid = np.stack(grid, axis=0)

    # Reshape grid to prepare for embedding calculation
    grid = grid.reshape([-1, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    # Add class token embedding if specified
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    # Add parameter tokens as zero embeddings
    param_tokens = np.zeros((num_param_tokens, embed_dim))
    pos_embed = np.concatenate([param_tokens, pos_embed], axis=0)

    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Compute 3D sine-cosine positional embeddings from a 3D grid.

    Args:
        embed_dim (int): Dimensionality of the embedding vector for each position.
        grid (numpy.ndarray): 3D grid of shape [3, grid_size, grid_size, grid_size].

    Returns:
        numpy.ndarray: Positional embeddings of shape [num_positions, embed_dim].
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even."

    # Divide dimensions equally for encoding grid_l, grid_h, and grid_w
    res = embed_dim // 3
    if res % 2 != 0:
        res += 1  # Ensure an even split
    factor_w = embed_dim - 2 * res

    # Compute positional embeddings for each axis
    emb_l = get_1d_sincos_pos_embed_from_grid(res, grid[0])  # (L*H*W, D//3)
    emb_h = get_1d_sincos_pos_embed_from_grid(res, grid[1])  # (L*H*W, D//3)
    emb_w = get_1d_sincos_pos_embed_from_grid(factor_w, grid[2])  # (L*H*W, remaining D)

    # Concatenate embeddings along the last dimension
    emb = np.concatenate([emb_l, emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Compute 1D sine-cosine positional embeddings for a sequence of positions.

    Args:
        embed_dim (int): Dimensionality of the embedding vector for each position.
        pos (numpy.ndarray): Sequence of positions to encode, of shape [num_positions].

    Returns:
        numpy.ndarray: Positional embeddings of shape [num_positions, embed_dim].
    """
    assert embed_dim % 2 == 0, "Embedding dimension must be even."

    # Compute the frequency range
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.0  # Normalize frequencies
    omega = 1.0 / (10000 ** omega)  # Compute scaled frequencies

    # Flatten positions and compute outer product with frequency
    pos = pos.reshape(-1)  # Flatten to [num_positions]
    out = np.einsum('m,d->md', pos, omega)  # Outer product: (num_positions, embed_dim / 2)

    # Compute sine and cosine embeddings
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    # Concatenate sine and cosine embeddings
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (num_positions, embed_dim)
    return emb
