# Data access and expected local paths

**Data are not distributed in this code-only repository.** Please obtain datasets from their original providers and comply with relevant access and licensing terms.

| Dataset | Source / documentation | Expected local file or package loader |
| --- | --- | --- |
| Adult | UCI Adult dataset; verify original URL | `data/adult/adult.data`, `data/adult/adult.test` |
| COMPAS | Public COMPAS dataset; verify original URL | `data/compas/compas-scores-two-years.csv` |
| Employee Salary | [Kaggle Employee Data (gmudit)](https://www.kaggle.com/datasets/gmudit/employer-data) | `data/Employers_data.csv` |
| Insurance | Paper reference: [Suess (2017) course material](http://www.sci.csueastbay.edu/~esuess/stat6620/#week-6) (currently inaccessible). Possible alternative: [Kaggle Medical Cost Personal Datasets](https://www.kaggle.com/datasets/mirichoi0218/insurance) (**file identity not yet verified**) | `data/insurance.csv` |
| FLCHAIN | `scikit-survival` | `sksurv.datasets.load_flchain()` |
| WHAS | `scikit-survival` | `sksurv.datasets.load_whas500()` |
| ROSSI | `lifelines` | `lifelines.datasets.load_rossi()` |

## Regression data

The Employee Salary dataset source was supplied by the authors. Download the original dataset and ensure the input CSV is named `Employers_data.csv` in the expected `data/` directory, after verifying its schema against the existing project preprocessing.

For Insurance, the manuscript cites Suess (2017). That reference URL is retained for bibliographic consistency but was not accessible at the time of this documentation update. A Kaggle listing with 1,338 observations and seven columns appears consistent with the paper's dataset description; **this is a candidate rather than a verified exact source**. Compare its schema/content to the original `insurance.csv` before use.

## Commands

The regression entry point accepts a dataset identifier and a data-file path. From `regression/`:

```bash
python run_regression_inproc_urw.py --dataset employers --data_path ../data/Employers_data.csv
python run_regression_inproc_urw.py --dataset insurance --data_path ../data/insurance.csv
```

The core scripts include task-specific preprocessing after loading. A verified automated downloader is not supplied. Commands were checked against argument parsing but have not been run-tested in the cleaned repository.

## Insurance input schema

The regression entry script `regression/run_regression_inproc_urw.py` expects the columns `age`, `sex`, `bmi`, `children`, `smoker`, `region`, and **`expenses`**. Some publicly shared Kaggle insurance CSVs call the outcome **`charges`** instead. Confirm that an externally downloaded file matches the original study dataset; rename the local `charges` column to `expenses` only after verification. No dataset files are bundled.

## Shell / working directory conventions

Run the documented commands while inside the named experiment folder. All paths beginning with `../data/` are relative to that working directory. Survival data are loaded through `scikit-survival` and `lifelines` and do not require CSV files in `data/`.
