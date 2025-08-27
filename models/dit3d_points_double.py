# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed

# import modules.functional as F
# from modules.voxelization import Voxelization


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(
            num_classes + use_cfg_embedding, hidden_size
        )
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = (
                torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
            )
        else:
            drop_ids = force_drop_ids == 1
        # print('token drop drop_ids:', drop_ids, drop_ids.shape)
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        # print('token drop labels:', labels, labels.shape)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat(
                    [
                        e,
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        e,
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                        e,
                    ]
                ),
            ]
        )
        self.register_buffer("basis", e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum("bnd,de->bne", input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings

    def forward(self, inp):
        # input: B x 3 x N
        input = inp.transpose(1, 2)  # B x N x 3
        emb = self.embed(input, self.basis)
        embed = self.mlp(torch.cat([emb, input], dim=2))  # B x N x C

        return embed.transpose(1, 2)  # B x C x N


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # approx_gelu = lambda: nn.GELU(approximate="tanh")
        approx_gelu = lambda: nn.GELU()  # for torch 1.7.1
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        num_points=2048,
        in_channels=6,
        hidden_size=512,  # 1152
        depth=8,  # 28
        num_heads=8,  # 16
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1,
        out_channels=3,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.num_points = num_points

        # self.x_embedder = PatchEmbed_Voxel(
        #     input_size, patch_size, in_channels, hidden_size, bias=True
        # )
        # num_patches = self.x_embedder.num_patches
        self.pos_embd = PointEmbed(dim=hidden_size)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, ref_cond, t, y, x_cond=None):
        """
        Forward pass of DiT.
        x: (N, 6, P) tensor of spatial inputs (point clouds or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        x, ref_pts = x[:, :3], x[:, 3:]
        # normalize the input x to [-1, 1]
        # x = x - x.mean(dim=2, keepdim=True)
        # x = x / (
        #     x.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values + 1e-6
        # ) # [-1, 1]

        # x = self.x_embedder(x)
        x_emb = self.pos_embd(x)

        inputs = x_emb + ref_cond
        if x_cond is not None:
            inputs = inputs + x_cond

        t = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t + y

        inputs = inputs.transpose(1, 2)

        for block in self.blocks:
            inputs = block(inputs, c)
        x = self.final_layer(inputs, c)

        return x.transpose(1, 2)

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    print("grid_size:", grid_size)

    grid_x = np.arange(grid_size, dtype=np.float32)
    grid_y = np.arange(grid_size, dtype=np.float32)
    grid_z = np.arange(grid_size, dtype=np.float32)

    grid = np.meshgrid(grid_x, grid_y, grid_z, indexing="ij")  # here y goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([3, 1, grid_size, grid_size, grid_size])
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 3 == 0

    # use half of dimensions to encode grid_h
    emb_x = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (X*Y*Z, D/3)
    emb_y = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (X*Y*Z, D/3)
    emb_z = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (X*Y*Z, D/3)

    emb = np.concatenate([emb_x, emb_y, emb_z], axis=1)  # (X*Y*Z, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL_2(pretrained=False, **kwargs):
    model = DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
    if pretrained:
        checkpoint = torch.load(
            "/path/to/DiT2D_pretrained_weights/DiT-XL-2-512x512.pt", map_location="cpu"
        )
        if "ema" in checkpoint:  # supports ema checkpoints
            checkpoint = checkpoint["ema"]
        checkpoint_blocks = {
            k: checkpoint[k] for k in checkpoint if not k.startswith("blocks")
        }
        # load pre-trained blocks from 2d DiT
        msg = model.load_state_dict(checkpoint_blocks, strict=False)

    return model


def DiT_XL_4(pretrained=False, **kwargs):
    model = DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)
    if pretrained:
        checkpoint = torch.load(
            "/path/to/DiT2D_pretrained_weights/DiT-XL-2-512x512.pt", map_location="cpu"
        )
        if "ema" in checkpoint:  # supports ema checkpoints
            checkpoint = checkpoint["ema"]
        checkpoint_blocks = {
            k: checkpoint[k] for k in checkpoint if k.startswith("blocks")
        }
        # load pre-trained blocks from 2d DiT
        msg = model.load_state_dict(checkpoint_blocks, strict=False)

    return model


def DiT_XL_8(pretrained=False, **kwargs):
    model = DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)
    if pretrained:
        checkpoint = torch.load(
            "/path/to/DiT2D_pretrained_weights/DiT-XL-2-512x512.pt", map_location="cpu"
        )
        if "ema" in checkpoint:  # supports ema checkpoints
            checkpoint = checkpoint["ema"]
        checkpoint_blocks = {
            k: checkpoint[k] for k in checkpoint if not k.startswith("blocks")
        }
        # load pre-trained blocks from 2d DiT
        msg = model.load_state_dict(checkpoint_blocks, strict=False)

    return model


def DiT_L_2(pretrained=False, **kwargs):
    return DiT(depth=24, hidden_size=1152, num_heads=16, **kwargs)


def DiT_L_4(pretrained=False, **kwargs):
    return DiT(depth=24, hidden_size=1152, num_heads=16, **kwargs)


def DiT_L_8(pretrained=False, **kwargs):
    return DiT(depth=24, hidden_size=1152, num_heads=16, **kwargs)


def DiT_B_2(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


def DiT_B_4(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


def DiT_B_8(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


def DiT_B_16(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


def DiT_B_32(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=768, num_heads=12, **kwargs)


def DiT_S_2(pretrained=False, **kwargs):
    return DiT(depth=12, hidden_size=384, num_heads=6, **kwargs)


def DiT_S_4(pretrained=False, **kwargs):
    model = DiT(depth=8, hidden_size=512, num_heads=8, **kwargs)

    return model


def DiT_S_8(pretrained=False, **kwargs):
    return DiT(depth=8, hidden_size=512, num_heads=8, **kwargs)


def DiT_S_16(pretrained=False, **kwargs):
    return DiT(depth=8, hidden_size=512, num_heads=8, **kwargs)


def DiT_S_32(pretrained=False, **kwargs):
    return DiT(depth=8, hidden_size=512, num_heads=8, **kwargs)


def DiT_tiny(pretrained=False, **kwargs):
    model = DiT(depth=8, hidden_size=512, num_heads=8, **kwargs)

    return model


def DiT_huge(pretrained=False, **kwargs):
    model = DiT(depth=24, hidden_size=512, num_heads=8, **kwargs)

    return model


DiT3D_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
    "DiT-tiny": DiT_tiny,
    "DiT-huge": DiT_huge,
}
