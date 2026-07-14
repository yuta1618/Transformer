import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        
        self.key = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.query = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.value = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)

        wei = (q @ k.transpose(-2, -1)) * (self.head_size**-0.5)
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = (wei @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.dropout(self.proj(out))