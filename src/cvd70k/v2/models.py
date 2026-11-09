"""Configurable named multiview network plus the official FT-Transformer."""

from __future__ import annotations

import torch
from torch import nn

MODULES = [
    "view:background",
    "view:haemodynamic",
    "view:anthropometric",
    "view:metabolic",
    "view:behaviour",
    "fusion1",
    "fusion2",
]


def block(a, b, drop=0.0):
    return nn.Sequential(nn.Linear(a, b), nn.ReLU(), nn.LayerNorm(b), nn.Dropout(drop))


class MultiView(nn.Module):
    def __init__(self, dims, cfg):
        super().__init__()
        self.dims = list(dims)
        e = cfg["embedding"]
        f1, f2 = cfg["fusion"]
        drop = cfg["dropout"]
        self.encoders = nn.ModuleList([block(d, e, drop) for d in dims])
        self.fusion1 = block(5 * e, f1, drop)
        self.fusion2 = block(f1, f2, drop)
        self.head = nn.Linear(f2, 1)

    def forward(self, x, representations=False):
        features = {}
        pieces = []
        for name, enc, view in zip(
            MODULES[:5], self.encoders, torch.split(x, self.dims, dim=1), strict=True
        ):
            z = enc(view)
            features[name] = z
            pieces.append(z)
        z = self.fusion1(torch.cat(pieces, 1))
        features["fusion1"] = z
        z = self.fusion2(z)
        features["fusion2"] = z
        logits = self.head(z).squeeze(-1)
        return (logits, features) if representations else logits

    def selectable_modules(self):
        return {
            **dict(zip(MODULES[:5], self.encoders, strict=True)),
            "fusion1": self.fusion1,
            "fusion2": self.fusion2,
        }


class MLP(nn.Module):
    def __init__(self, inputs, cfg):
        super().__init__()
        layers = []
        for width in cfg["hidden"]:
            layers.append(block(inputs, width, cfg["dropout"]))
            inputs = width
        self.net = nn.Sequential(*layers, nn.Linear(inputs, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class FT(nn.Module):
    """Official rtdl feature tokenizer/backbone, with native categorical embeddings.

    Fold-scaled base features are inverted using ONLY fitted train mean/scale.
    This reconstructs schema-defined categorical codes without learning categories.
    """

    def __init__(self, prep, cfg):
        super().__init__()
        from rtdl_revisiting_models import FTTransformer

        names = prep.names
        self.cat_names = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
        cat_encoded = {
            "gender",
            "smoke",
            "alco",
            "active",
            "cholesterol_1",
            "cholesterol_2",
            "cholesterol_3",
            "gluc_1",
            "gluc_2",
            "gluc_3",
        }
        self.cont = [i for i, c in enumerate(names) if c not in cat_encoded]
        self.positions = {
            c: [names.index(c)] if c in names else [names.index(f"{c}_{j}") for j in [1, 2, 3]]
            for c in self.cat_names
        }
        self.register_buffer(
            "mean",
            torch.tensor(
                prep.scaler.mean_ if hasattr(prep.scaler, "mean_") else prep.scaler.center_,
                dtype=torch.float32,
            ),
        )
        self.register_buffer("scale", torch.tensor(prep.scaler.scale_, dtype=torch.float32))
        self.model = FTTransformer(
            n_cont_features=len(self.cont),
            cat_cardinalities=[2, 3, 3, 2, 2, 2],
            d_out=1,
            n_blocks=cfg["blocks"],
            d_block=cfg["token_width"],
            attention_n_heads=4,
            attention_dropout=cfg["dropout"],
            ffn_d_hidden=None,
            ffn_d_hidden_multiplier=4 / 3,
            ffn_dropout=cfg["dropout"],
            residual_dropout=0.0,
        )

    def forward(self, x):
        raw = x * self.scale + self.mean
        cats = []
        for name in self.cat_names:
            indices = self.positions[name]
            cat = (
                raw[:, indices].argmax(1)
                if len(indices) == 3
                else raw[:, indices[0]].round().long().clamp(0, 1)
            )
            cats.append(cat)
        return self.model(x[:, self.cont], torch.stack(cats, 1)).squeeze(-1)


def model_for(prep, cfg):
    if cfg["family"] == "multiview":
        return MultiView(prep.dims, cfg)
    if cfg["family"] == "mlp":
        return MLP(len(prep.names), cfg)
    if cfg["family"] == "ft_transformer":
        return FT(prep, cfg)
    raise ValueError(cfg["family"])


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
