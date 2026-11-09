"""Source-distance and constrained radial adaptation, independent of V2 artifacts."""

import copy

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from cvd70k.v2.models import MODULES
from cvd70k.v2.selector import calibrated_mask, matched_random, set_mask


def module_names(model):
    mapping = {}
    parameters = dict(model.named_parameters())
    for module, block in model.selectable_modules().items():
        ids = {id(p) for p in block.parameters()}
        mapping[module] = [n for n, p in parameters.items() if id(p) in ids]
    return mapping


def anchors(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def source_penalty(model, reference, weights):
    mapping = module_names(model)
    loss = next(model.parameters()).new_zeros(())
    for m, names in mapping.items():
        for n in names:
            if dict(model.named_parameters())[n].requires_grad:
                loss = (
                    loss
                    + weights[m]
                    * (
                        (
                            dict(model.named_parameters())[n]
                            - reference[n].to(dict(model.named_parameters())[n])
                        )
                        ** 2
                    ).sum()
                )
    return loss


def scores(evidence):
    caps = {}
    geo = {}
    mmd = {}
    for m, r in evidence.items():
        valid = r["geometry_status"] == "valid"
        u = r["U"] if valid else 1.0
        rr = r["R"] if valid else 1.0
        p = r["domain_p"] if r["domain_status"] == "valid" else 1.0
        degradation = float(np.clip(1 - u, 0, 1))
        retained = float(np.clip(1 - rr, 0, 1))
        caps[m] = (
            float(0.1 + 0.8 * (0.7 * degradation + 0.3 * (1 - p) * retained)) if valid else 0.0
        )
        geo[m] = 0.01 + 0.99 * (1 - degradation) if valid else 1.0
        mmd[m] = 0.01 + 0.99 * p
    return caps, geo, mmd


class Radial(nn.Module):
    """No independent trainable gate: radius of the effective delta identifies it."""

    def __init__(self, source, caps):
        super().__init__()
        self.base = copy.deepcopy(source)
        self.mapping = module_names(self.base)
        self.caps = dict(caps)
        self.radii = {}
        self.deltas = nn.ParameterDict()
        self.parameter_order = list(dict(self.base.named_parameters()))
        for p in self.base.parameters():
            p.requires_grad = False
        for p in self.base.head.parameters():
            p.requires_grad = True
        for i, m in enumerate(MODULES):
            values = torch.cat(
                [dict(self.base.named_parameters())[n].detach().flatten() for n in self.mapping[m]]
            )
            self.radii[m] = 0.1 * max(float(values.norm()), 1.0)
            self.deltas[str(i)] = nn.Parameter(
                torch.zeros_like(values), requires_grad=bool(caps[m] > 0)
            )

    def effective(self):
        params = dict(self.base.named_parameters())
        for i, m in enumerate(MODULES):
            offset = 0
            delta = self.deltas[str(i)] * self.caps[m]
            for n in self.mapping[m]:
                p = params[n]
                params[n] = p + delta[offset : offset + p.numel()].reshape_as(p)
                offset += p.numel()
        return params

    def forward(self, x):
        return functional_call(self.base, self.effective(), (x,))

    def project(self):
        with torch.no_grad():
            for i, m in enumerate(MODULES):
                d = self.deltas[str(i)]
                d.mul_(torch.clamp(self.radii[m] / d.norm().clamp_min(1e-12), max=1.0))

    def regularizer(self):
        return 0.01 * sum(
            (self.caps[m] * self.deltas[str(i)]).square().sum() for i, m in enumerate(MODULES)
        )

    def materialize(self):
        model = copy.deepcopy(self.base)
        model.load_state_dict(
            {**model.state_dict(), **{n: p.detach() for n, p in self.effective().items()}}
        )
        return model

    def gates(self):
        return {
            m: dict(
                cap=self.caps[m],
                radial=float(self.deltas[str(i)].detach().norm()) / self.radii[m],
                effective_gate=self.caps[m]
                * float(self.deltas[str(i)].detach().norm())
                / self.radii[m],
            )
            for i, m in enumerate(MODULES)
        }


def policy(source, name, evidence, seed):
    logical, _ = calibrated_mask(evidence)
    allmask = {m: True for m in MODULES}
    mask = allmask.copy()
    weights = {m: 0.0 for m in MODULES}
    gates = None
    caps, geo, mmd = scores(evidence)
    if name == "head":
        mask = {m: False for m in MODULES}
    elif name in ["logical", "logical_sp_low", "logical_sp_high"]:
        mask = logical
    elif name == "random_matched":
        mask, _ = matched_random(source, logical, seed)
    single = {
        "view_hemo": MODULES[1],
        "view_metabolic": MODULES[3],
        "view_background": MODULES[0],
        "view_anthro": MODULES[2],
        "view_behaviour": MODULES[4],
        "fusion1_only": MODULES[5],
        "fusion2_only": MODULES[6],
    }
    if name in single:
        mask = {m: m == single[name] for m in MODULES}
    if name == "fusion_both":
        mask = {m: m in MODULES[5:] for m in MODULES}
    if "_sp_" in name:
        weights = {m: 0.01 if name.endswith("low") else 1.0 for m in MODULES}
    if name.startswith("reg_"):
        weights = (
            geo
            if name == "reg_geometry"
            else mmd
            if name == "reg_mmd"
            else {m: (geo[m] + mmd[m]) / 2 for m in MODULES}
        )
        if name == "reg_shuffled":
            weights = dict(
                zip(
                    MODULES,
                    np.random.default_rng(seed).permutation(list(weights.values())),
                    strict=True,
                )
            )
    model = copy.deepcopy(source)
    set_mask(model, mask)
    if name.startswith("soft_"):
        if name == "soft_uniform":
            caps = {m: 0.5 for m in MODULES}
        if name == "soft_learned":
            caps = {m: 1.0 for m in MODULES}
        if name == "soft_random":
            caps = dict(zip(MODULES, np.random.default_rng(seed).uniform(0.1, 0.9, 7), strict=True))
        if name == "soft_shuffled":
            caps = dict(
                zip(
                    MODULES,
                    np.random.default_rng(seed).permutation(list(caps.values())),
                    strict=True,
                )
            )
        model = Radial(source, caps)
        gates = caps
        mask = {m: caps[m] > 0 for m in MODULES}
    return model, mask, weights, gates


def drift(source, target):
    sm = dict(source.named_parameters())
    tm = dict(target.named_parameters())
    result = {}
    for m, names in module_names(source).items():
        delta = torch.cat([(tm[n].detach().cpu() - sm[n].detach().cpu()).flatten() for n in names])
        ref = torch.cat([sm[n].detach().cpu().flatten() for n in names])
        result[m] = {
            "absolute": float(delta.norm()),
            "relative": float(delta.norm() / ref.norm().clamp_min(1e-12)),
        }
    return result
