"""Vision Transformer (ViT): the deliberate non-convolutional contrast
baseline in this repo -- no `nn.Conv2d` anywhere.

Reference: Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zhai, Unterthiner,
Dehghani, Minderer, Heigold, Gelly, Uszkoreit, Houlsby, "An Image is Worth
16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021.
arXiv:2010.11929. See papers/README.md (bibtex key `dosovitskiy2021vit`).

Every other model in this repo is built from `nn.Conv2d`, which gives a
convolution's inductive bias for free: nearby pixels are processed
together, and the same small filter is reused at every spatial position.
ViT removes that assumption entirely: an image is cut into fixed-size
patches, each patch is linearly embedded (just an `nn.Linear`, exactly like
a word embedding), a learned position embedding is added so the model can
recover *some* notion of "where" a patch came from (nothing forces it to
learn locality the way a conv kernel's receptive field does), and the
whole sequence is processed by a plain Transformer encoder -- the same
architecture used for text. Multi-head self-attention is hand-written
below (`nn.Linear` Q/K/V projections + softmax), not `nn.MultiheadAttention`.

Comparing ViT's CIFAR-10-from-scratch accuracy against any CNN in this
repo (see ../../docs/source/model_comparison.md) is a direct empirical
answer to "what does convolution's inductive bias actually buy you." The
paper itself is explicit that ViT needs much more pretraining data than a
CNN to reach comparable accuracy -- trained from scratch here on CIFAR-10
alone with no large-scale pretraining, at these epoch budgets, it is
*expected* to trail the CNNs in this repo. That gap is the finding, not a
bug to apologize for.

Kept small for CPU training speed: patch size 4 (64 patches for a 32x32
image), embedding dim 128, 4 encoder blocks, 4 attention heads -- far
smaller than the paper's own configurations.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def image_to_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """x: (B, C, H, W) -> (B, num_patches, C*patch_size*patch_size), each
    patch flattened in (C, p, p) order. Verified (see model dev notes) to
    exactly invert back to the original image with a matching reassembly."""
    b, c, h, w = x.shape
    p = patch_size
    x = x.unfold(2, p, p).unfold(3, p, p)  # (B, C, H/p, W/p, p, p)
    x = x.permute(0, 2, 3, 1, 4, 5).contiguous()  # (B, H/p, W/p, C, p, p)
    return x.view(b, (h // p) * (w // p), c * p * p)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, n_heads, N, head_dim)
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class EncoderBlock(nn.Module):
    """Pre-LN Transformer block: x = x + Attn(LN(x)); x = x + MLP(LN(x))."""

    def __init__(self, dim: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTModel(nn.Module):
    """(B, 3, 32, 32) -> (B, num_classes). No convolution anywhere."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 128,
        depth: int = 4,
        n_heads: int = 4,
        num_classes: int = 10,
    ):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.patch_embed = nn.Linear(patch_dim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([EncoderBlock(embed_dim, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = image_to_patches(x, self.patch_size)  # (B, num_patches, patch_dim)
        tokens = self.patch_embed(patches)  # (B, num_patches, embed_dim)

        b = tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, num_patches+1, embed_dim)
        tokens = tokens + self.pos_embed

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        cls_out = tokens[:, 0]  # the [CLS] token's final representation
        return self.head(cls_out)
