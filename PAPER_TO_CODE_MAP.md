# Paper-to-Code Map

**Manuscript:** *Individual Fairness via Local Predictive Consistency* (TMLR camera-ready preparation)  
**Repository:** `local-predictive-consistency`  
**Status:** Static correspondence audit, 13 October 2026. **Not an executed reproduction test.**

## How to read this map
- **Located** = a script or pipeline visibly implements the associated experiment or analysis; not proof of exact published configuration.
- **Candidate** = appears relevant based on script name and/or CSV input/output; exact figure/table construction requires checking the published source files and settings.
- **Manual/theory** = paper prose, proof, or analytic construction; there may be no standalone training script.
- Data files and generated results are deliberately excluded from the code-only public repository. Figure scripts that read existing CSV files cannot reproduce a figure until those CSVs are regenerated.

## Main paper

| Paper item | Dataset(s) / content | Relevant repository files | Status and reproduction notes |
|---|---|---|---|
| Table 1 | Adult, COMPAS, Employee Salary, Insurance, FLCHAIN, WHAS, ROSSI; sizes/features | `data/README.md`, `employee_data_cleaning.py`; data loaders within `binary/run_binary_inproc_urw_adult.py`, `regression/run_regression_inproc_urw.py`, `survival/run_survival_inproc_deepsurv_urw.py` | Located at dataset level; table statistics require confirming preprocessed files and cohort sizes |
| Figure 1 | Binary classification performance vs violation rate across λ | `binary/run_binary_inproc_urw_adult.py`, `binary/run_binary_inproc_lap.py`, `binary/plot_tradeoff_balacc_vr.py`, `binary/tradeoff_analysis.py` | Located training; plotting candidates. Exact exported panels and λ grids unverified |
| Table 2 | Adult balanced accuracy and VR, including SenSR, SenSeI, iFlipper, Laplacian, URW-FKL/RKL | `binary/run_binary_inproc_urw_adult.py`, `binary/run_binary_inproc_lap.py`, `binary/run_ifair.py`, `binary/run_iflip.py` | **Partial**: third-party baseline source removed; SenSR/SenSeI dependencies/results not fully packaged; selected validation-matched operating points need explicit mapping |
| Figure 2 | Regression RMSE versus VR on Employee Salary and Insurance | `regression/run_regression_inproc_urw.py`, `regression/run_regression_inproc_laplacian.py`, `regression/trade_off_regression.py`, `regression/tradeoff_vr.py` | Located training; plotting candidates. Dataset names/column schemas and λ sweep to verify |
| Figure 3 | Survival IBS/C-index vs VR across FLCHAIN/WHAS/ROSSI; DeepSurv and survival-function | `survival/run_survival_inproc_deepsurv_urw.py`, `survival/run_survival_inproc_deepsurv_laplacian.py`, `survival_discrete/run_survival_inproc_urw.py`, `survival_discrete/run_survival_inproc_laplacian.py`, `results/trade_off_survival.py`, `survival_discrete/trade_off_analysis.py` | Located training; candidate visualization. Original CSV dependencies omitted |

## Appendix

| Paper item | Subject | Relevant repository files | Status and open issue |
|---|---|---|---|
| A | Implementation details | Four main URW training scripts, corresponding Laplacian scripts, launch scripts (`*.sh`), `requirements.txt` | Located broadly; exact package versions/seeds/settings per reported run not verified |
| B, Table 3 | Independent pairwise fairness: EdgeMeanAbs and related metrics | `revise/compute_pairwise_folder_metrics.py`, `revise/pairwise_edge_fairness_final.py` | Located diagnostics; CSV naming and all tasks / selected λ points need verification |
| C, Tables 4–7 | Graph coverage, isolated nodes and τ sensitivity (Adult, Employee Salary, WHAS) | `revise/graph_coverage.py`, `revise/binary_coverage.py`, `revise/conditional_if_covered_analysis.py`, `revise/isolated_node_correction.py`, `revise/survival_isolation.py`, `binary/tau_sensitivity.py`, `regression/tau_sensitivity.py`, `survival/tau_sensitivity.py`, `binary/run_binary_inproc_urw_bin_unified_tau_sensitivity.py`, `regression/run_regression_sensitivity.py`, `survival/run_sensitivity_survival.py` | Located multiple related pipelines; per-table selections and scripts' file expectations require verification |
| D, Table 8 | Controlled cancellation examples | No uniquely identifiable dedicated script | **Manual/theory / unverified**: document source or calculation for controlled values |
| E, Table 9 | Pareto points evaluated on achieved pairwise fairness | `revise/pareto_all_tasks.py`, `revise/compare_regression_laplacian_urw.py` | Located analytical candidates; must trace specific intermediate metrics |
| F, Figures 4–5 | GR-consistency and S-consistency on binary datasets | `binary/consistancy_analysis.py`, `binary/lamda_analysis.py`, `binary/tradeoff_analysis.py` | **Candidate** based on analysis names; verify exact graph/score computation and output figures |
| G, Figure 6 | p-norm / tail sensitivity | `regression/run_regression_p_norm_urw.py`, `regression/analysis_pnorm.py`, `regression/pnorm_figure.py`, `regression/pnorm_summary.py`, `binary/analysis_pnorm_compare.py`, `survival/run_survival_inproc_deepsurv_urw_pnorm.py`, `survival/run_survival_inproc_deepsurv_laplacian_pnorm.py` | Located p-norm experiments, candidate presentation scripts; specific published panels unverified |
| H, Table 10 | Multi-seed robustness on small survival datasets | `survival/run_deepsurv_multiseed.sh`, `survival_discrete/run_curve_multiseed.sh`, related DeepSurv/curve training scripts | Located launch candidates; must confirm seeds and summary aggregation |
| I, Tables 11–12 | URW vs combinatorial and symmetric normalized Laplacian | `binary/run_binary_inproc_lap.py`, `binary/run_binary_inproc_nlap.py`, `binary/run_adult_inproc_nrw.py`, `regression/run_regression_inproc_laplacian.py`, `regression/run_regression_inproc_nrw.py`, `survival/run_survival_inproc_deepsurv_laplacian.py`, `survival/run_survival_inproc_deepsurv_nrw.py`, `revise/compare_regression_laplacian_urw.py` | Located comparable method families; paper operating points not cross-checked |
| J, Tables 13–16 | Waller-style consistency Adult/COMPAS, k=5/10/15, δ=.5, disagreement cases | `revise/compute_waller_binary_metrics.py`, `binary/find_instances.py` | Located diagnostics; instance CSV inputs were omitted from code-only repo and need regeneration |
| K | Proof of Proposition 1 | Paper appendix; no code needed | Manual/theory |
| L | Forward- and Reverse-KL barycenter derivations | Paper appendix; binary implementation `binary/run_binary_inproc_urw_adult.py` | Theory with corresponding implementation located |
| M, Figure 7 | Sensitivity to neural-network depth | `depth_sensitivity.py`, `regression/compare_depth.py`; training scripts expose `--depth` | Located analysis and configurable training; exact plotted inputs unverified |

## Main experiment entry points

Run **from the corresponding subdirectory**, with input datasets provisioned as described in `data/README.md` / `DATA_ACCESS_NOTES.md`:

```bash
cd binary
python run_binary_inproc_urw_adult.py --dataset adult --data_dir ../data --agg forward
# substitute --agg reverse for the Reverse-KL variant

cd ../regression
python run_regression_inproc_urw.py --dataset employers

cd ../survival
python run_survival_inproc_deepsurv_urw.py --dataset whas

cd ../survival_discrete
python run_survival_inproc_urw.py --dataset whas
```

These are **illustrative commands with arguments confirmed by static code inspection**, not independently tested exact commands for reproducing the paper's figures. A complete reproduction requires checking dataset locations, preprocessing, exact λ/τ/seeds/architecture, dependencies and outputs against the final manuscript.
