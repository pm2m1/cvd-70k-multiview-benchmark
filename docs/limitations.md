# Limitations and interpretation

- This is cross-sectional internal label classification. It provides no prediction of
  incident cardiovascular events, future time-bounded risk, clinical validity or deployment
  readiness. Label ascertainment/adjudication and recruitment population are poorly documented.
- Kaggle's official metadata lists licence **Unknown**. Original acquisition source, consent,
  clinical institution, country/site, detailed measurement procedures and row-to-person linkage
  are not verified. No participant data, actual example records, fitted models or predictions
  are publicly redistributed. See provenance/release documentation before downstream reuse.
- Five seeds reuse one held-out sample. Seed SD is not external-validation uncertainty.
  Feature-group bootstrap handles exact duplicate dependence only; the ID is not a verified
  participant ID. Repeated people with different recorded features could remain undetected.
- Feature-identical rows with discordant labels remain ambiguous. No method can resolve that
  ambiguity solely from these recorded inputs. Duplicates are retained with isolated groups.
- Plausibility bounds are broad analyst choices, not validated diagnostic rules or guaranteed
  impossibility limits. Minimal masking/imputation may hide informative missingness; strict
  exclusion changes case mix and evaluation prevalence. Strict and minimal AP are not directly
  paired estimates on the same test population. No threshold was optimized on test performance.
- BMI is deterministic given its parents. Its redundancy changes representation and capacity;
  BMI-only and parent-only controls explore this without claiming independent information.
- A modest equal two-candidate tuning budget may miss stronger settings in every family. The
  fixed configuration is reused in low-prevalence conditions; no model gets additional tuning.
  LayerNorm flat MLP and multiview are close in parameter count but differ in depth/topology.
- Prevalence manipulation also changes available support. Fixed total trades positive for
  negative support; fixed positives changes total/negative support. The available pool forces
  a low common positive count in B. These experiments do not identify a pure causal effect of
  prevalence. Sampling and optimization vary together under paired seeds in imbalance runs.
- Validation/test retain approximately 50% positives. Weighting/oversampling counter an
  artificial training-prior shift and can improve recall/calibration there. That is not
  evidence of benefit at a 7.42%-prevalence deployment population. No real MHAS domain shift
  is reproduced, and AP should not be compared naively with the historical 7.42% MHAS test.
- Thresholds maximize validation balanced accuracy, without clinical harm/benefit or treatment
  utility. Threshold grids are descriptive. No deployment recommendation follows.
- Primary and secondary comparisons, many imbalance conditions and sensitivities are evaluated
  on one fixed test after a single lock. Descriptive bootstrap intervals are not adjusted for
  multiplicity and do not establish universal multiview superiority. Model/seed averaging in
  the primary statistic averages AP, not probabilities; no unregistered ensemble is the winner.
- No genuine acquisition sites or independent cohorts are established. No selective adaptation,
  simulated shift or external transfer experiment is manufactured. These are future work only.
