from __future__ import annotations
import os, sys, warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, f1_score, roc_auc_score
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
sns.set_style("whitegrid")

# 0. Paths
load_dotenv()
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
PLOTS_DIR   = os.getenv("PLOTS_DIR",   "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

csv_path = os.path.join(RESULTS_DIR, "stream_scores.csv")
if not os.path.isfile(csv_path):
    sys.exit("❌  run stream_models.py first – stream_scores.csv not found.")

df = pd.read_csv(csv_path)
needed = {"label", "if_score", "fusion_prob"}
missing = needed.difference(df.columns)
if missing:
    sys.exit(f"❌  stream_scores.csv missing: {', '.join(missing)}")

# 1. ROC curve
curves = {}
for name, col in {"IForest":"if_score",
                  "Fusion" :"fusion_prob"}.items():
    fpr, tpr, _ = roc_curve(df["label"], df[col])
    curves[name] = (fpr, tpr, auc(fpr, tpr))

plt.figure(figsize=(5,5))
for name,(fpr,tpr,auc_) in curves.items():
    plt.plot(fpr, tpr, label=f"{name}  AUC={auc_:.2f}", lw=2)
plt.plot([0,1],[0,1],"k--", lw=1)
plt.xlabel("False-Positive Rate"); plt.ylabel("True-Positive Rate")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "roc.png"), dpi=300)
print("✓  plots/roc.png written")

# 2. Latency bar (optional)
lat_txt = os.path.join(RESULTS_DIR, "latency_ms.txt")
if os.path.isfile(lat_txt):
    vals = np.loadtxt(lat_txt)
    if vals.size == 3:
        plt.figure()
        plt.bar(["IForest","Fusion-total","Fusion"], vals)
        plt.ylabel("ms per record"); plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "latency.png"), dpi=300)
        print("✓  plots/latency.png written")

# 3. Console summary
auc_if  = curves["IForest"][2]
auc_fus = curves["Fusion"][2]

thr_if  = np.percentile(df["if_score"], 99)       # 1 % contamination
f1_if   = f1_score(df["label"], df["if_score"]>thr_if)

best_thr = 0.40  # already printed by stream_models.py
f1_fus  = f1_score(df["label"], df["fusion_prob"]>best_thr)

print("\nAUC  IF={:.2f}  Fusion={:.2f}".format(auc_if, auc_fus))
print("F1   IF={:.2f}  Fusion={:.2f}".format(f1_if, f1_fus))
