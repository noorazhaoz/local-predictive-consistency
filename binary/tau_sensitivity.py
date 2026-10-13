import pandas as pd

path = "results_unified/tau_sensitivity_adult_usemetric.csv"
df = pd.read_csv(path)

cols = [
    "DATASET",
    "agg_mode",
    "tau_quantile",
    "tau_used",
    "graph_train_avg_degree",
    "graph_train_isolated_rate",
    "graph_train_edge_density",
    "test_bal_acc",
    "test_auc",
    "IF_mean(method,KL,prob)",
    "IF_median(method,KL,prob)",
]

summary = df[cols].copy()

summary["tau_used"] = summary["tau_used"].round(4)
summary["graph_train_avg_degree"] = summary["graph_train_avg_degree"].round(2)
summary["graph_train_isolated_rate"] = summary["graph_train_isolated_rate"].round(4)
summary["graph_train_edge_density"] = summary["graph_train_edge_density"].round(4)
summary["test_bal_acc"] = summary["test_bal_acc"].round(4)
summary["test_auc"] = summary["test_auc"].round(4)
summary["IF_mean(method,KL,prob)"] = summary["IF_mean(method,KL,prob)"].round(6)
summary["IF_median(method,KL,prob)"] = summary["IF_median(method,KL,prob)"].round(6)

print(summary)

summary.to_csv("tau_sensitivity_summary_adult.csv", index=False)

