import torch
import torch.nn as nn


class SCRSeqEncoder(nn.Module):
    def __init__(self, in_dim: int = 2, hidden_dim: int = 64, num_layers: int = 1, out_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.gru = nn.GRU(
            input_size=int(in_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=(float(dropout) if int(num_layers) > 1 else 0.0),
        )
        self.proj = nn.Sequential(
            nn.Linear(int(hidden_dim), int(out_dim)),
            nn.LayerNorm(int(out_dim)),
            nn.GELU(),
        )

    def forward(self, seq_x: torch.Tensor, seq_mask: torch.Tensor):
        # seq_x: [B, L, 2], seq_mask: [B, L]
        lengths = seq_mask.sum(dim=1).long()
        out = torch.zeros(seq_x.shape[0], self.hidden_dim, device=seq_x.device, dtype=seq_x.dtype)

        has = lengths > 0
        if has.any():
            xh = seq_x[has]
            lh = lengths[has]
            lh_sorted, idx = torch.sort(lh, descending=True)
            x_sorted = xh[idx]

            packed = nn.utils.rnn.pack_padded_sequence(x_sorted, lh_sorted.cpu(), batch_first=True, enforce_sorted=True)
            _, h_n = self.gru(packed)
            last = h_n[-1]

            inv = torch.empty_like(idx)
            inv[idx] = torch.arange(idx.numel(), device=idx.device)
            last = last[inv]
            out[has] = last

        return self.proj(out)
