# Transfer feasibility and endpoint gate

## A. Recorded age subgroups — feasible, internal only

The authorized original V1 training pool has sufficient recorded age variation and positives
for the age<55→age≥55 design. Seven roles and exact counts are in
`results/v2/aggregate/transfer_feasibility_counts.json`. The target fitting pool has 5,320
positives/3,446 negatives; a disjoint selector pool has 3,989/2,583. Both support every
registered 25–1,000-positive, 50%-prevalence budget and the 250-positive/750-negative control.
Target evaluation is fixed at4,383 records,2,660 positives; stopping uses2,191 records,
including1,330 positives. That stopping-label cost is large and must accompany every budget.
Source pretraining has21,669 rows; source stopping and selector-reference roles are separate.
Age is retained as a predictor, and restricted age ranges create extrapolation/case-mix
limitations. Feature-identical groups are disjoint, but actual person linkage is unverified.

## B. Simulated domain shift — engineering controls only

Synthetic matrices with known mean shift, direction reversal, constant features and sparse
classes verify MMD, geometry, abstention and freezing. They are test fixtures, not dataset
performance. A new artificial dataset-shift performance study is not needed to manufacture
external evidence; the real recorded age diagnostic proceeds independently.

## C. External cohorts — no eligible pair established

[Official UCI documentation](https://archive.ics.uci.edu/dataset/45/heart+disease) describes
angiographic coronary narrowing, unlike the unspecified broad Kaggle `cardio` diagnosis.
The official ZIP was retrieved only for feasibility; raw files remain private. Actual site
counts: Cleveland303 (139 positive), Hungary294 (106), Switzerland123 (115), VA Long Beach200
(149), using num>0. Missing feature cells and hashes are recorded in
`external_feasibility.json`. The processed14-column schema lacks complete height/weight/BMI,
diastolic BP and behaviour views. Numeric cholesterol and fasting glucose thresholds also
do not supply a verified crosswalk to Kaggle's categories. Site identity is real, but that
does not harmonize endpoints/features or supply sufficient external target support.

[NHANES 2021–2023 official MCQ codebook](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/MCQ_L.htm)
reports332 yes responses for ever-diagnosed heart attack and367 for ever stroke. These are
marginal questionnaire frequencies, **not** verified composite-positive counts or an eligible
merged analytic cohort. Self-report, age eligibility, examination/laboratory missingness and
complex survey sampling require a new harmonization protocol. No NHANES participant data
were downloaded or used for training.

[HAALSI/HAALSA official access](https://haalsi.org/haalsa-data/) distributes data through
NACDA/ICPSR. [MHAS documentation](https://www.mhasweb.org/Documentation/Home.aspx) and its
[2021 questionnaire](https://www.mhasweb.org/resources/DOCUMENTS/2021/Core_Questionnaire_2021.pdf)
separate doctor-reported heart attack from stroke/TIA and other heart conditions. No private
Phase1 data/checkpoints were accessed. Exact contemporary eligible counts and a Kaggle label
crosswalk are not verified. The historical incident/reporting design cannot be silently
reused as a cross-sectional Kaggle endpoint.

[CRELES official documentation](https://ccp.ucr.ac.cr/creles/documentacion/) provides wave-specific
questionnaires; verified access, diagnosis semantics, overlap, measurement crosswalk and
eligible positive counts are not established here. Those counts are unknown, not zero.

Kaggle clinical ascertainment remains insufficiently defined, so no external pair passes the
full gate: identity, permissions, endpoint/time horizon, baseline covariates, outcome rules,
positive counts, missingness, label budgets, test isolation, overlap and calibration population.
**EXTERNAL_CROSS_COHORT_TRANSFER_NOT_EXECUTED**. This is an eligibility conclusion, not a
negative transfer-performance result. No external model fit or external clinical claim exists.
