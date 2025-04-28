from __future__ import annotations
import os, time, warnings
from typing import List

import numpy  as np
import pandas as pd
from tqdm            import tqdm
from dotenv          import load_dotenv
from sklearn.ensemble import IsolationForest
from xgboost          import XGBClassifier
from sklearn.metrics  import (roc_auc_score, f1_score,
                              precision_recall_curve)

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# 0. paths & data
load_dotenv()
CLEAN_DIR   = os.getenv("CLEAN_DIR",   "data/clean_data")
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

df = pd.read_csv(f"{CLEAN_DIR}/soil_stream_features.csv", parse_dates=["ts"])
if "label" not in df.columns:
    raise RuntimeError(
        "❌  'label' missing.  Run inject_anomalies.py & build_features.py first."
    )

df = df[["ts", "sensor_id",
         "moisture_z", "delta_z", "neighbor_delta", "label"]]

split        = int(0.60 * len(df))
train, test  = (df.iloc[:split].reset_index(drop=True),
                df.iloc[split:].reset_index(drop=True))

# 1. models
WIN = 1_440           # 1-day rolling window   (60 s cadence × 24 h)
buf: List[List[float]] = []
ifor = IsolationForest(n_estimators=100,
                       contamination=0.01,
                       random_state=0)

def step_iforest(v: list[float]) -> float:
    buf.append(v); buf[:] = buf[-WIN:]
    if len(buf) < WIN:
        return 0.0
    ifor.fit(buf)
    return float(-ifor.decision_function([v])[0])

# 2. stream TRAIN
X_tr, y_tr = [], []
for r in tqdm(train.itertuples(index=False), total=len(train), desc="TRAIN"):
    iscore = step_iforest([r.moisture_z, r.delta_z, r.neighbor_delta])
    X_tr.append([iscore, r.moisture_z, r.neighbor_delta])
    y_tr.append(r.label)

fusion = XGBClassifier(
    n_estimators = 60,
    max_depth    = 4,
    learning_rate= 0.10,
    eval_metric  = "logloss",
    verbosity    = 0,
)
fusion.fit(np.asarray(X_tr), y_tr)

p, r_, thr = precision_recall_curve(y_tr, fusion.predict_proba(X_tr)[:,1])
best_thr   = float(thr[(2*p*r_/(p+r_+1e-9)).argmax()])
print(f"Optimal fusion threshold ≈ {best_thr:.2f}")

# 3. stream TEST
if_s, fus_s = [], []
tic = time.perf_counter()
for r in tqdm(test.itertuples(index=False), total=len(test), desc="TEST"):
    iscore = step_iforest([r.moisture_z, r.delta_z, r.neighbor_delta])
    prob   = fusion.predict_proba([[iscore, r.moisture_z,
                                    r.neighbor_delta]])[0,1]
    if_s.append(iscore); fus_s.append(prob)
lat_ms = 1_000 * (time.perf_counter() - tic) / len(test)

# 4. save & metrics
out = test.copy()
out["if_score"]    = if_s
out["fusion_prob"] = fus_s
out.to_csv(f"{RESULTS_DIR}/stream_scores.csv", index=False)
np.savetxt(f"{RESULTS_DIR}/latency_ms.txt",
           np.round([lat_ms/2, lat_ms/2, lat_ms], 2)) 

auc_if  = roc_auc_score(out.label, out.if_score)
auc_fus = roc_auc_score(out.label, out.fusion_prob)

f1_if   = f1_score(out.label,
                   out.if_score > np.percentile(out.if_score, 99))
f1_fus  = f1_score(out.label, out.fusion_prob > best_thr)

print(f"\nAUC   IF={auc_if:.2f}  Fusion={auc_fus:.2f}")
print(f"F1    IF={f1_if:.2f}  Fusion={f1_fus:.2f}")
print(f"Avg latency ≈ {lat_ms:.2f} ms / row")
