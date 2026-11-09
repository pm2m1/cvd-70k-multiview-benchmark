"""Small multiview mechanisms with explicit matched controls."""

import torch
from torch import nn

from cvd70k.v2.models import FT, MLP, MultiView, block


class Advanced(MultiView):
    def __init__(self, dims, cfg):
        super().__init__(dims, cfg)
        self.kind = cfg["architecture"]
        e = cfg["embedding"]
        h = cfg["fusion"][0]
        if self.kind in ["gated", "uniform", "random"]:
            values = torch.zeros(5)
            if self.kind == "random":
                values = torch.logit(0.2 + 0.6 * torch.rand(5))
            self.gates = nn.Parameter(values, requires_grad=self.kind == "gated")
        if self.kind in ["residual", "depth_control"]:
            self.residual = block(h, h, cfg["dropout"])
        if self.kind.startswith("attention"):
            self.identity = nn.Parameter(torch.zeros(5, e), requires_grad=self.kind == "attention")
            if self.kind == "attention":
                nn.init.normal_(self.identity, std=0.02)
            self.attention = nn.MultiheadAttention(e, 4, batch_first=True, dropout=cfg["dropout"])
            self.attention_norm = nn.LayerNorm(e)
        if self.kind == "pairwise":
            self.fusion1 = block(5 * e + 10, h, cfg["dropout"])

    def forward(self, x):
        z = torch.stack(
            [
                enc(view)
                for enc, view in zip(self.encoders, torch.split(x, self.dims, 1), strict=True)
            ],
            1,
        )
        if hasattr(self, "gates"):
            z = z * self.gates.sigmoid()[None, :, None]
        if hasattr(self, "attention"):
            tokens = z + self.identity[None]
            att, _ = self.attention(tokens, tokens, tokens, need_weights=False)
            z = self.attention_norm(tokens + att)
        flat = z.flatten(1)
        if self.kind == "pairwise":
            pairs = torch.stack(
                [(z[:, i] * z[:, j]).mean(1) for i in range(5) for j in range(i + 1, 5)], 1
            )
            flat = torch.cat([flat, pairs], 1)
        h = self.fusion1(flat)
        if hasattr(self, "residual"):
            extra = self.residual(h)
            h = extra + h if self.kind == "residual" else extra
        return self.head(self.fusion2(h)).squeeze(-1)


def total(model):
    return sum(p.numel() for p in model.parameters())


def build(prep, cfg):
    kind = cfg["architecture"]
    if kind == "ft":
        return FT(prep, cfg)
    if kind == "concat":
        return MultiView(prep.dims, cfg)
    if kind == "flat":
        reference = Advanced(prep.dims, {**cfg, "architecture": cfg["match"]})
        target = total(reference)
        options = []
        for a in range(32, 193, 8):
            for b in range(16, 129, 8):
                n = len(prep.names) * a + 3 * a + a * b + 3 * b + b + 1
                options.append((abs(n - target), a, b))
        _, a, b = min(options)
        return MLP(len(prep.names), {**cfg, "hidden": [a, b]})
    return Advanced(prep.dims, cfg)
