import pandas as pd

path = "results/tau_sensitivity_employers_usemetric.csv"
df = pd.read_csv(path)

cols = [
    "dataset",
    "tau_quantile",
    "tau_used",
    "graph_train_avg_degree",
    "graph_train_isolated_rate",
    "graph_train_edge_density",
    "test_RMSE_real",
    "test_MAE_real",
    "test_R2_real",
    "IF_mean(URW_L1,log_pred)",
    "IF_median(URW_L1,log_pred)",
]

summary = df[cols].copy()

summary["tau_used"] = summary["tau_used"].round(4)
summary["graph_train_avg_degree"] = summary["graph_train_avg_degree"].round(2)
summary["graph_train_isolated_rate"] = summary["graph_train_isolated_rate"].round(4)
summary["graph_train_edge_density"] = summary["graph_train_edge_density"].round(4)

summary["test_RMSE_real"] = summary["test_RMSE_real"].round(4)
summary["test_MAE_real"] = summary["test_MAE_real"].round(4)
summary["test_R2_real"] = summary["test_R2_real"].round(4)

summary["IF_mean(URW_L1,log_pred)"] = summary["IF_mean(URW_L1,log_pred)"].round(6)
summary["IF_median(URW_L1,log_pred)"] = summary["IF_median(URW_L1,log_pred)"].round(6)

print(summary)

summary.to_csv("tau_sensitivity_summary_employers.csv", index=False)