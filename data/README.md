# Data access and preparation

This repository contains research code but does not redistribute raw datasets. Download the required data from the sources below and follow the providers' terms of use.

## Datasets

| Task | Dataset | Source | Expected location or loader |
| --- | --- | --- | --- |
| Binary classification | Adult | UCI Adult dataset | `data/adult/adult.data`, `data/adult/adult.test` |
| Binary classification | COMPAS | Public COMPAS dataset | `data/compas/compas-scores-two-years.csv` |
| Regression | Employee Salary | [Kaggle Employee Data (gmudit)](https://www.kaggle.com/datasets/gmudit/employer-data) | `data/Employers_data.csv` |
| Regression | Insurance | [Suess (2017), cited in the paper](http://www.sci.csueastbay.edu/~esuess/stat6620/#week-6); [alternative Kaggle listing](https://www.kaggle.com/datasets/mirichoi0218/insurance) | `data/insurance.csv` |
| Survival analysis | FLCHAIN | `scikit-survival` | `sksurv.datasets.load_flchain()` |
| Survival analysis | WHAS | `scikit-survival` | `sksurv.datasets.load_whas500()` |
| Survival analysis | ROSSI | `lifelines` | `lifelines.datasets.load_rossi()` |

The alternative Insurance listing has not been confirmed to be identical to the file used in the experiments. Check its content and column names against the original data before substituting it.

## File structure

Place downloaded classification and regression files under `data/` as follows:

```text
data/
├── adult/
│   ├── adult.data
│   └── adult.test
├── compas/
│   └── compas-scores-two-years.csv
├── Employers_data.csv
└── insurance.csv
```

Survival datasets are loaded by the corresponding Python packages and do not require local CSV files.

## Regression commands

Run these commands from the `regression/` directory:

```bash
python run_regression_inproc_urw.py --dataset employers --data_path ../data/Employers_data.csv
python run_regression_inproc_urw.py --dataset insurance --data_path ../data/insurance.csv
```

The scripts perform task-specific preprocessing after loading the data. Paths beginning with `../data/` are relative to the experiment directory.

### Insurance data schema

The Insurance implementation uses `expenses` as its response variable and expects the following columns:

`age`, `sex`, `bmi`, `children`, `smoker`, `region`, `expenses`.

Some externally available versions use `charges` instead of `expenses`. If obtaining an alternative version, first confirm that it corresponds to the study data and then match the expected input schema. **The original experimental file uses `expenses`; no changes to the research code are needed for that file.**

## Reproducibility note

Input data are not included in this code-only release. The listed commands reflect the script interfaces, but the cleaned public repository has not been independently tested end-to-end on a fresh installation.
