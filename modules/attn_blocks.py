import inspect
import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


"""
Attentions, all we need
"""


class AttentionBase(nn.Module):
    def __init__(self, n_embd, n_head, bias, dropout, **kwargs):
        super().__init__()
        assert n_embd % n_head == 0
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")


class CrossAttention(AttentionBase):
    def __init__(self, n_embd, n_head, bias, dropout):
        super().__init__(n_embd, n_head, bias, dropout)
        self.ckv_attn = nn.Linear(n_embd, 2 * n_embd, bias=bias)
        self.cq_attn = nn.Linear(n_embd, n_embd, bias=bias)

    def forward(self, x, encoding, kv_cache=None, mask=None):
        B, T_q, C = x.size()
        _, T_kv, C_kv = encoding.size()
        q = self.cq_attn(x)
        q = q.view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2)

        if kv_cache is not None and kv_cache.shape[1] > 0:
            k, v = kv_cache.unbind(dim=0)
            mask = mask[:, :, :, :T_kv]  # make sure the shape matches
        else:
            k, v = self.ckv_attn(encoding).split(self.n_embd, dim=2)
            k = k.view(B, T_kv, self.n_head, C // self.n_head).transpose(1, 2)
            v = v.view(B, T_kv, self.n_head, C // self.n_head).transpose(1, 2)

        # Efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,  # the encoded feature is global, no need to mask
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,  # False?
        )

        y = y.transpose(1, 2).contiguous().view(B, T_q, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        # Update the cache with the current k, v projections
        return y, None if kv_cache is None else torch.stack((k, v))


class QueryAttention(AttentionBase):
    def __init__(self, config):
        super().__init__(config)
        self.cq_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.ckv_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)

    def forward(self, query, encoding, kv_mask=None):
        B, T_q, C = query.size()
        _, T_kv, C_kv = encoding.size()
        q = self.cq_attn(query)
        q = q.view(B, T_q, self.n_head, C // self.n_head).transpose(1, 2)

        k, v = self.ckv_attn(encoding).split(self.n_embd, dim=2)
        k = k.view(B, T_kv, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T_kv, self.n_head, C // self.n_head).transpose(1, 2)

        if kv_mask is not None:
            # (B, n_head, T_q, T_kv)
            kv_mask = kv_mask.view(B, 1, 1, T_kv).expand(-1, self.n_head, T_q, -1)
            # note that kv_mask should be a boolean mask

        else:
            kv_mask = None

        # Efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=kv_mask,  # the encoded feature is global, no need to mask
            dropout_p=self.dropout if self.training else 0,
            is_causal=False,
        )

        y = y.transpose(1, 2).contiguous().view(B, T_q, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))

        # Update the cache with the current k, v projections
        return y


class QuerySelfAttention(AttentionBase):
    def __init__(self, config):
        super().__init__(config)
        # key, query, value projections for all heads, but in a batch
        self.qk_attn = nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
        self.v_attn = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, qk, v):
        (
            B,
            T,
            C,
        ) = qk.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k = self.qk_attn(qk).split(self.n_embd, dim=2)
        v = self.v_attn(v)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y


"""
blocks used (w/ attention)
"""


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, kv_cache=None, mask=None):
        out, kv_cache = self.attn(self.ln_1(x), kv_cache, mask)
        x = x + out
        x = x + self.mlp(self.ln_2(x))
        return x, kv_cache


class BlockWithCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.xattn = CrossAttention(config)
        self.mlp = MLP(config)

    def forward(self, x, encoding, kv_cache_sa=None, kv_cache_ca=None, mask=None):
        out, kv_cache_sa = self.attn(self.ln_1(x), kv_cache_sa, mask)
        x = x + out
        out, kv_cache_ca = self.xattn(self.ln_2(x), encoding, kv_cache_ca, mask)
        x = x + out
        x = x + self.mlp(self.ln_3(x))
        return x, kv_cache_sa, kv_cache_ca


class BlockWithQueryAttentionDETR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.sattn = QuerySelfAttention(config)
        self.xattn = QueryAttention(config)
        self.mlp = MLP(config)

    def forward(self, kv, tgt, query_pos):
        tgt2 = self.ln_1(tgt)
        tgt2 = self.sattn(qk=tgt2 + query_pos, v=tgt2)
        tgt = tgt + tgt2
        tgt2 = self.ln_2(tgt)
        tgt2 = self.xattn(tgt2 + query_pos, kv)
        tgt = tgt2 + tgt
        tgt2 = self.ln_3(tgt)

        tgt = tgt + self.mlp(tgt2)

        return tgt


class BlockWithSrcAttentionDETR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.sattn = QuerySelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, src):
        src2 = self.ln_1(src)
        src2 = self.sattn(qk=src2, v=src2)
        src = src + src2
        src2 = self.ln_2(src)

        src = src + self.mlp(src2)

        return src


class BlockCrossAttnOnly(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.xattn = QueryAttention(config)
        self.mlp = MLP(config)

    def forward(self, kv, tgt, query_pos, kv_mask=None):
        tgt2 = self.ln_1(tgt)
        tgt2 = self.xattn(tgt2 + query_pos, kv, kv_mask=kv_mask)
        tgt = tgt2 + tgt
        tgt2 = self.ln_2(tgt)
        tgt = tgt + self.mlp(tgt2)

        return tgt


"""
all the networks
"""


def configure_optimizers(
    named_parameters,
    weight_decay,
    learning_rate,
    betas,
    device_type,
    additional_params=None,
):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in named_parameters}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    if additional_params is not None:
        if isinstance(additional_params, dict):
            for n, p in additional_params:
                nodecay_params.append(p)
        else:
            for additional_param in additional_params:
                for n, p in additional_param:
                    nodecay_params.append(p)
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    print(f"using fused AdamW: {use_fused}")

    return optimizer
