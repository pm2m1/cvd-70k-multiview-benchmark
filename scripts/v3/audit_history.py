"""Read-only historical development diagnostics; no V1/V2 prediction execution."""

from pathlib import Path

import pandas as pd

from cvd70k.utils.io import read_json, sha256, write_json
from cvd70k.v2.selector import calibrated_mask
from cvd70k.v3.design import PUB, preserve

preserve()
rows = []
checkpoints = []
for p in Path("results/v2/aggregate/transfer_trials").glob("*.json"):
    r = read_json(p)
    t = r["training"]
    c = t["curve"]
    rows.append(
        dict(
            run=p.stem,
            best_epoch=t["best_epoch"],
            epochs_completed=t["epochs_completed"],
            source_ap=c[0]["validation_ap"],
            retained_ap=c[t["best_epoch"]]["validation_ap"],
            validation_ap_change=c[t["best_epoch"]]["validation_ap"] - c[0]["validation_ap"],
            epoch_zero=t["best_epoch"] == 0,
            fit_seconds=t["fit_seconds"],
        )
    )
    path = Path("private/v2/transfer_models") / (p.stem + ".joblib")
    if path.exists():
        if sha256(path) != r["checkpoint_sha256"]:
            raise ValueError("Historical checkpoint changed")
        checkpoints.append(r["checkpoint_sha256"])
frame = pd.DataFrame(rows)
frame.to_csv(PUB / "v2_adaptation_checkpoint_diagnostics.csv", index=False)
evidence = read_json("results/v2/aggregate/transfer_evidence.json")
checked = 0
for r in evidence:
    mask, _ = calibrated_mask(r["modules"])
    if mask != r["logical_mask"]:
        raise ValueError("V2 selector aggregate reproduction failed")
    checked += 1
write_json(
    PUB / "v2_reference_verification.json",
    dict(
        aggregate_masks_verified=checked,
        historical_checkpoint_hashes_verified=len(checkpoints),
        checkpoint_digest=__import__("hashlib")
        .sha256("".join(sorted(checkpoints)).encode())
        .hexdigest(),
        adaptation_records=len(frame),
        epoch_zero_fraction=float(frame.epoch_zero.mean()),
        median_best_epoch=float(frame.best_epoch.median()),
        median_stop_ap_change=float(frame.validation_ap_change.median()),
        diagnostics_scope="Saved target stopping records only; no historical evaluation predictions opened",
    ),
)
text = f"""# V2 failure analysis and V3 motivation

Historical V2 development audit reproduced all {checked} published calibrated-logical masks using the unchanged policy. It inspected {len(frame)} saved adaptation training records and verified {len(checkpoints)} available checkpoint hashes without inference. Epoch-zero retention fraction was {frame.epoch_zero.mean():.3f}; median best epoch {frame.best_epoch.median():.1f}. Median retained-minus-source stopping AP was {frame.validation_ap_change.median():+.5f}. These are stopping-set improvements subject to checkpoint selection bias, not unbiased adaptation gains. See `results/v3/aggregate/v2_adaptation_checkpoint_diagnostics.csv`.

V2's reported source-only ranking was strong and logical selection did not beat head-only or label-cost-matched full fine-tuning at 1,000 fitting positives. These frozen findings motivate source-preservation and smaller stopping-set studies; they do not identify a causal failure mechanism. More ADAPT decisions and mask agreement measured geometry stability, not predictive benefit. V2 used 2,191 stopping labels (1,330 positive) in addition to fitting and selector labels; V3 explicitly charges all target roles.

Potential causes are hypotheses: binary freezing may be too coarse; representation shift may not predict useful adaptation; checkpoint noise and target-label allocation may dominate tiny effects; source representations may already transfer adequately. V3 tests constrained drift and geometry-weighted penalties with development-only single-module counterfactuals. Correlations between evidence and module benefit use only the dedicated V3 selection role. No outer result redesigns a selector. Source-only, full/head/random and target-only references remain necessary.

MMD sample caps remain 128 in the exact reference; geometry uses the full selector budget. Higher support can narrow geometry uncertainty while leaving capped MMD noisy. The learned-meta-selector is infeasible: three overlapping age tasks do not supply independent meta-training populations. This prevents fitting a selector that memorizes a small collection of episodes.

V2 historical performance is context; V3 uses different internal roles and cannot be ranked against historical numbers as though the populations were matched. V1/V2 configurations are refitted on the V3 classification roles for direct comparison. No external cohort or future-event validity follows.
"""
Path("docs/v3/V2_FAILURE_ANALYSIS.md").write_text(text)
Path("docs/v3/TRANSFER_FAILURE_DIAGNOSIS.md").write_text(text)
print("Historical development audit complete", len(frame), checked)
