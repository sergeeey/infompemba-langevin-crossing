"""Print the reference numbers of the configurations suggested to an outside reviewer (REVIEW_REQUEST.md, task 3).

Reads prereg_v2/summary_main.csv only (no curves needed). The configurations are chosen by a fixed rule written here,
not by looking at which ones agree: the first row of each (b, T, kappa) listed below, plus the first W1 NO_TEST row.
"""

import pandas as pd

CHOSEN = [(1.0, 0.2, 0.0), (2.0, 0.2, 0.0), (1.0, 0.1, 0.1), (0.1, 1.0, 0.1), (3.0, 0.3, 0.05)]
COLS = [
    "b",
    "T",
    "kappa",
    "lam1",
    "lam_src",
    "kl_label",
    "kl_tstar",
    "kl_p0",
    "w1_label",
    "w1_tstar",
    "w1_p0",
    "c1_cold",
    "c1_hot",
    "r",
]

M = pd.read_csv("prereg_v2/summary_main.csv")
rows = [M[(M.b == b) & (M["T"] == t) & (M.kappa == k)].iloc[0] for b, t, k in CHOSEN]
rows.append(M[M.w1_label == "NO_TEST"].iloc[0])
pd.set_option(
    "display.width", 250, "display.max_columns", 30, "display.float_format", "{:.6g}".format
)
print(pd.DataFrame(rows)[COLS].to_string(index=False))
