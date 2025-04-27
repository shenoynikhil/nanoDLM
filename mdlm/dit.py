"""Simple implementation fo a bert style encoder only network

Implemented with
- Rotary Embeddings (RoPE)
- Diffusion Transformer (DiT) from https://arxiv.org/abs/2212.09748
"""

import torch
import torch.nn as nn
from torch import Tensor

from einops import rearrange


def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def apply_rope(x: Tensor, pos_emb: Tensor) -> Tensor:
    xeven, xodd = x[..., 0::2], x[..., 1::2] # x{even/odd}:[batch_size, seq_len, n_heads, head_dim // 2]
    cos_emb, sin_emb = pos_emb[..., :xeven.shape[-1]], pos_emb[..., xeven.shape[-1]:]
    return torch.cat((
        xeven * cos_emb - xodd * sin_emb,
        xeven * sin_emb + xodd * cos_emb
    ), dim=-1) # [batch_size, seq_len, n_heads, hidden_dim]


class RoPE(nn.Module):
    """Rotary Positional Embedding"""
    def __init__(self, hidden_dim: int):
        super().__init__()

        # get the frequencies, 10000^(-2i/d),
        dim_half = hidden_dim // 2
        self.freqs = 1.0 / (10000 ** (torch.arange(0, dim_half, 1).float() / dim_half)) # [hidden_dim // 2]

        # cache the cos and sin values for a given sequence length
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x: Tensor,seq_dim: int = 1) -> Tensor:
        """x: [batch_size, seq_len, n_heads, head_dim]"""
        # Add the positional embedding to the input tensor.
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached: # If length has changed, compute new freqs and cache values
            t = torch.arange(seq_len, device=x.device) # [seq_len]
            self.freqs = self.freqs.to(x.device) # move to correct device
            angles = t[:, None] * self.freqs[None, :] # [seq_len, hidden_dim // 2]
            self.cos_cached = torch.cos(angles)[None, :, None, :] # [1, seq_len, 1, hidden_dim // 2]
            self.sin_cached = torch.sin(angles)[None, :, None, :] # [1, seq_len, 1, hidden_dim // 2]
            self.seq_len_cached = seq_len # update the cached sequence length

        # apply the rotation to the even and odd indices
        pos_emb = torch.cat([self.cos_cached, self.sin_cached], dim=-1) # [1, seq_len, 1, hidden_dim]
        return pos_emb


class Attention(nn.Module):
    """Multi-Head Self Attention with RoPE"""
    def __init__(self, hidden_dim: int, n_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.qkv_bias = qkv_bias
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=qkv_bias)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: Tensor, pos_emb: Tensor) -> Tensor:
        qkv = self.qkv.forward(x) # [batch_size, seq_len, hidden_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch, seq_len, hidden_dim]
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.n_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.n_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.n_heads)
        q = apply_rope(q, pos_emb)
        k = apply_rope(k, pos_emb)

        # set shape for scaled_dot_product_attention
        q = rearrange(q, 'b s h d -> b h s d')
        k = rearrange(k, 'b s h d -> b h s d')
        v = rearrange(v, 'b s h d -> b h s d')

        # apply scaled dot product attention, rearrange and project
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, 'b h s d -> b s (h d)')
        x = self.proj(x)

        return x # [batch_size, seq_len, hidden_dim]


class AdaLNZeroBlock(nn.Module):
    """AdaLNZeroBlock: Ada(ptive) L(ayer) N(orm) Zero(initialized) Block
    From the DiT paper. Basically a linear layer to learn the shift and scale weights.
    Both weight and bias are initialized to zero.
    """
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)        


class DiTBlock(nn.Module):
    """DiTBlock"""
    def __init__(self, hidden_dim: int, n_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = True):
        super().__init__()
        self.norm_msa = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_dim, n_heads, qkv_bias)
        self.norm_mlp = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim)
        )
        self.adaLN = AdaLNZeroBlock(input_dim=hidden_dim, output_dim=6 * hidden_dim)
    
    def forward(self, x: Tensor, c: Tensor, pos_emb: Tensor) -> Tensor:
        # x(input): [batch_size, seq_len, hidden_dim], c(onditioning): [batch_size, hidden_dim]
        # Following DiT Block with adaLN-Zero conditioning (figure 3, page 3), we need scale, shift
        # and gate factors for msa and mlp, msa: multi-head self-attention, mlp: pointwise feed-forward network
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN.forward(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm_msa(x), shift_msa, scale_msa), pos_emb)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm_mlp(x), shift_mlp, scale_mlp))
        return x


class DiTFinalLayer(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.adaLN = AdaLNZeroBlock(input_dim=hidden_dim, output_dim=2 * hidden_dim)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x = self.norm_final(x)
        shift_final, scale_final = self.adaLN.forward(c).chunk(2, dim=1)
        x = modulate(x, shift_final, scale_final)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """DiT: Diffusion Transformer
    
    Note: This is not exactly the same DiT as in the MDLM (Sahoo et al, 2024) paper.
    """
    def __init__(
        self,
        vocab_size: int,
        n_layers: int = 2,
        n_heads: int = 2,
        hidden_dim: int = 64,
        qkv_bias: bool = True,
    ):
        super().__init__()

        # embedding, positional encoding, block layers, and output layer
        self.vocab_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.noise_level_embedding = nn.Linear(1, hidden_dim)
        self.rope = RoPE(hidden_dim // n_heads)
        self.layers = nn.ModuleList([DiTBlock(hidden_dim, n_heads, qkv_bias=qkv_bias) for _ in range(n_layers)])
        self.output_layer = DiTFinalLayer(hidden_dim, vocab_size)
    
    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        x = self.vocab_embedding(x) # [batch_size, seq_len] -> [batch_size, seq_len, hidden_dim]
        c = self.noise_level_embedding(c) # [batch_size,] -> [batch_size, hidden_dim]
        pos_emb = self.rope(x, seq_dim=1)
        for layer in self.layers:
            x = layer(x, c, pos_emb)
        logits = self.output_layer(x, c)
        return logits # [batch_size, seq_len, vocab_size]


if __name__ == "__main__":
    # Simple test for DiT forward pass
    vocab_size = 100
    batch_size = 2
    seq_len = 8
    hidden_dim = 32
    n_heads = 4  # Make sure hidden_dim % n_heads == 0 and (hidden_dim // n_heads) is even!
    n_layers = 2

    # Create random input tokens and conditioning vector
    x = torch.randint(0, vocab_size, (batch_size, seq_len))  # [batch_size, seq_len]
    c = torch.randn(batch_size, 1)                  # [batch_size,]

    # Instantiate the models
    model = DiT(
        vocab_size=vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        hidden_dim=hidden_dim,
        qkv_bias=True
    )

    # Forward pass
    logits = model(x, c)  # [batch_size, seq_len, vocab_size]
    print("Logits shape:", logits.shape)
    assert logits.shape == (batch_size, seq_len, vocab_size), "Output shape mismatch!"
    print("Forward pass successful!")
