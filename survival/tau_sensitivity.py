import pandas as pd

path = "results/urw/whas/test_tau_sensitivity_whas_usemetric.csv"
df = pd.read_csv(path)

cols = [
    "dataset",
    "tau_quantile",
    "tau_used",
    "graph_train_avg_degree",
    "graph_train_isolated_rate",
    "graph_train_edge_density",
    "test_c_index",
    "brier_mean",
    "ibs",
    "IF_mean(|(I-P)eta|,test)",
    "IF_median(|(I-P)eta|,test)",
]

summary = df[cols].copy()

summary["tau_used"] = summary["tau_used"].round(4)
summary["graph_train_avg_degree"] = summary["graph_train_avg_degree"].round(2)
summary["graph_train_isolated_rate"] = summary["graph_train_isolated_rate"].round(4)
summary["graph_train_edge_density"] = summary["graph_train_edge_density"].round(4)

summary["test_c_index"] = summary["test_c_index"].round(4)
summary["brier_mean"] = summary["brier_mean"].round(4)
summary["ibs"] = summary["ibs"].round(4)

summary["IF_mean(|(I-P)eta|,test)"] = summary["IF_mean(|(I-P)eta|,test)"].round(6)
summary["IF_median(|(I-P)eta|,test)"] = summary["IF_median(|(I-P)eta|,test)"].round(6)

print(summary)

summary.to_csv("tau_sensitivity_summary_whas.csv", index=False)