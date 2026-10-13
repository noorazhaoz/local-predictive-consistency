# Individual Fairness via Local Predictive Consistency

Code accompanying the manuscript **Individual Fairness via Local Predictive Consistency**, accepted with minor revision at *Transactions on Machine Learning Research (TMLR)*.

**Authors:** Zhouting Zhao and Tin Lok James Ng  
**Affiliation:** Trinity College Dublin  
**Repository:** https://github.com/noorazhaoz/local-predictive-consistency

## Overview

We study **local predictive consistency** as a node-level relaxation of pairwise individual fairness. For each instance, the method compares its prediction with a similarity-weighted average of neighboring predictions. The resulting local residual provides an individual-level diagnostic and motivates a fairness regularizer based on the row-random-walk graph Laplacian (URW).

The experiments cover binary classification, regression, and time-to-event prediction. We evaluate predictive performance and local consistency against graph-Laplacian and other individual-fairness baselines.

## Repository structure

| Directory | Contents |
| --- | --- |
| `binary/` | Adult and COMPAS classification; URW Forward-KL and Reverse-KL; comparison and analysis scripts |
| `regression/` | Regression experiments, Laplacian comparisons, p-norm and sensitivity analyses |
| `survival/` | DeepSurv / log-risk experiments |
| `survival_discrete/` | Survival-function prediction experiments |
| `revise/` | Graph coverage, pairwise diagnostics, Waller-style consistency and other robustness analyses |
| `results/` | Plotting and analysis scripts (generated outputs are not included) |
| `data/` | Data access and preparation notes |

See [`PAPER_TO_CODE_MAP.md`](PAPER_TO_CODE_MAP.md) for a guide to the code associated with the manuscript's figures, tables and appendices.

## Installation

The experiments were developed on **macOS using Python 3.8** and run primarily as standalone Python scripts. To create a local environment:

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency list provides a starting point; exact package versions from the original environment have not been recovered.

## Running the main experiments

Run commands from the respective experiment directories because some scripts use relative paths. Download the required datasets as described in [`data/README.md`](data/README.md).

**Binary classification (Adult, Forward-KL):**

```bash
cd binary
python run_binary_inproc_urw_adult.py --dataset adult --data_dir ../data --agg forward
```

**Regression (Employee Salary):**

```bash
cd regression
python run_regression_inproc_urw.py --dataset employers --data_path ../data/Employers_data.csv
```

**Survival analysis (DeepSurv, WHAS):**

```bash
cd survival
python run_survival_inproc_deepsurv_urw.py --dataset whas
```

**Survival-function prediction (WHAS):**

```bash
cd survival_discrete
python run_survival_inproc_urw.py --dataset whas
```

Each command assumes you start in the repository root. Other datasets, aggregation methods and experiment configurations can be selected using the scripts' available arguments. The commands identify the main entry points; they have not been independently tested in a fresh environment after repository curation. Some figures and additional analyses require locally generated intermediate outputs.

## Datasets

This is a **code-only** repository. Dataset files, trained models and generated experimental results are not redistributed.

| Task | Dataset | Source / loading method |
| --- | --- | --- |
| Classification | Adult | [UCI Adult](https://archive.ics.uci.edu/dataset/2/adult); expected files: `data/adult/adult.data`, `data/adult/adult.test` |
| Classification | COMPAS | [ProPublica COMPAS data](https://github.com/propublica/compas-analysis); expected file: `data/compas/compas-scores-two-years.csv` |
| Regression | Employee Salary | [Kaggle Employee Data](https://www.kaggle.com/datasets/gmudit/employer-data); local path: `data/Employers_data.csv` |
| Regression | Insurance | [Suess (2017), cited in the paper](http://www.sci.csueastbay.edu/~esuess/stat6620/#week-6); local path: `data/insurance.csv` |
| Survival | FLCHAIN | `sksurv.datasets.load_flchain()` |
| Survival | WHAS | `sksurv.datasets.load_whas500()` |
| Survival | ROSSI | `lifelines.datasets.load_rossi()` |

The Insurance experiments use an input file whose outcome column is named **`expenses`**. When obtaining a file from a different source, check its column names and data compatibility before running the scripts. The manuscript's cited Insurance source is retained above; it is not presented as a verified direct CSV download.

For local file layout, preprocessing and additional data notes, see [`data/README.md`](data/README.md).

## Reproducibility and external baselines

The repository includes the authors' core implementation and associated analysis scripts. Third-party baseline implementations, including iFair, iFlipper and SenSR, are not redistributed; any wrappers that depend on them require the corresponding upstream code and dependencies. Generated intermediate files must be recreated locally for plotting or downstream analyses. Exact reproduction of every reported result has not been independently verified from a clean installation.

## License

No license has been selected for the original code yet. Permission for reuse will be governed by a license added after agreement between the authors. Third-party code and data remain subject to their own terms.

## Citation

Zhouting Zhao and Tin Lok James Ng. *Individual Fairness via Local Predictive Consistency*. Transactions on Machine Learning Research (TMLR), accepted with minor revision (2026). Please cite the final published article when its bibliographic details become available.
