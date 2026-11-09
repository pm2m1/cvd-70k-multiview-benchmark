# Local dataset only

Obtain version 1 of [Svetlana Ulianova's Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset)
through Kaggle after reviewing its terms. Official metadata says **Unknown** licence.
There is no verified permission here to redistribute participant data. No raw data,
assignments, predictions, fitted preprocessing or model weights are released.

Run `uv run python scripts/prepare.py --archive '/path/to/archive(1).zip'` from the project
root. The archive is inspected before extraction. `raw/` and `processed/` are ignored.
This project does not download a substitute dataset. The original download timestamp and
upstream clinical acquisition institution/consent are not established by the supplied ZIP.
