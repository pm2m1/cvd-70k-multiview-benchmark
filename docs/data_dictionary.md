# Data dictionary

Verified from the supplied CSV schema and [official Kaggle metadata](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).

|Field|Unit / encoding|Role|
|---|---|---|
|id|Unique integer row identifier|Excluded from predictors; not verified person ID|
|age|Days; converted using /365.25|Background|
|gender|Observed codes 1,2; semantic mapping unverified|Background, binary 0,1 transform|
|height|cm|Anthropometric|
|weight|kg|Anthropometric|
|ap_hi|Systolic BP, assumed conventional mmHg|Haemodynamic|
|ap_lo|Diastolic BP, assumed conventional mmHg|Haemodynamic|
|cholesterol|1 normal, 2 above normal, 3 well above normal|Metabolic, three one-hot inputs|
|gluc|1 normal, 2 above normal, 3 well above normal|Metabolic, three one-hot inputs|
|smoke|0/1, self-report smoking|Behaviour|
|alco|0/1, self-report alcohol intake|Behaviour|
|active|0/1, self-report physical activity|Behaviour|
|cardio|0 absence / 1 presence of cardiovascular disease|Cross-sectional target|
|age_years|Derived continuous years|Replaces age days in models|
|bmi|Derived weight/(height/100)², kg/m²|Anthropometric, valid parents only|

The official page supplies no diagnostic adjudication, time horizon, validated site labels,
detailed recruitment frame, measurement protocol or affirmative redistribution licence.
Missing units for numeric BP are interpreted conventionally and remain a provenance limit.
