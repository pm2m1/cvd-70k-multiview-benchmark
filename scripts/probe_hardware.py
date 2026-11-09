"""Actual forward/backward parity and tiny-network accelerator timing."""

import copy
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from cvd70k.models.networks import FiveViewMLP
from cvd70k.training.engine import set_seed
from cvd70k.utils.io import write_json

if __name__ == "__main__":
    torch.set_num_threads(2)
    set_seed(99)
    x, y = torch.randn(512, 16), torch.randint(0, 2, (512,)).float()
    base = FiveViewMLP()
    result = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "torch": torch.__version__,
        "cpu_threads": torch.get_num_threads(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "devices": {},
    }
    for key in ["machdep.cpu.brand_string", "hw.memsize"]:
        p = subprocess.run(["sysctl", "-n", key], capture_output=True, text=True)
        result[key] = p.stdout.strip() if p.returncode == 0 else "unavailable in sandbox"
    for device in ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []):
        try:
            net = copy.deepcopy(base).to(device)
            xx, yy = x.to(device), y.to(device)
            net.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xx), yy)
            loss.backward()
            gradients = torch.cat([p.grad.flatten().cpu() for p in net.parameters()]).numpy()
            if device == "cpu":
                reference = gradients
            err = float(np.max(np.abs(gradients - reference)))
            start = time.perf_counter()
            opt = torch.optim.AdamW(net.parameters())
            for _ in range(100):
                opt.zero_grad()
                z = torch.nn.functional.binary_cross_entropy_with_logits(net(xx), yy)
                z.backward()
                opt.step()
            if device == "mps":
                torch.mps.synchronize()
            result["devices"][device] = {
                "seconds_100_steps": time.perf_counter() - start,
                "gradient_max_abs_error_vs_cpu": err,
                "finite": bool(torch.isfinite(z)),
                "reliable": bool(torch.isfinite(z)) and err < 1e-4,
            }
        except Exception as e:
            result["devices"][device] = {"error": repr(e), "reliable": False}
    candidates = {
        k: v["seconds_100_steps"] for k, v in result["devices"].items() if v.get("reliable")
    }
    result["fastest_device_in_probe"] = min(candidates, key=candidates.get)
    result["selected_device"] = "mps" if "mps" in candidates else "cpu"
    result["selection_reason"] = (
        "Use reliable MPS when available as requested; CPU is faster for this tiny probe."
    )
    write_json("results/aggregate/hardware.json", result)
    print(result)
