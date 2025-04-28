import os, pickle, numpy as np, pandas as pd
from dotenv import load_dotenv

load_dotenv()
CLEAN_DIR = os.getenv("CLEAN_DIR")

df = pd.read_csv(f"{CLEAN_DIR}/soil_stream_native.csv", parse_dates=["ts"])

# 1. Start with gap_flag
df["label"] = df["gap_flag"].astype(int)

# 2. +-0.20 absolute spike/drop on 1 % of rows per sensor
np.random.seed(42)
def inject(g):
    n = int(0.01 * len(g))
    if n == 0:
        return g
    idx    = np.random.choice(g.index, n, replace=False)
    signs  = np.random.choice([-1, 1], n)
    g.loc[idx, "moisture"] = (g.loc[idx, "moisture"] + signs*0.20).round(2)
    g.loc[idx, "label"] = 1
    return g

df = df.groupby("sensor_id", group_keys=False).apply(inject)

# 3. Save
out_file = f"{CLEAN_DIR}/soil_stream_labeled.csv"
df.to_csv(out_file, index=False)
pickle.dump(df.index[df.label == 1], open(f"{CLEAN_DIR}/anom_idx.pkl", "wb"))

print(f"Total rows: {len(df):,}")
print(f"Total anomalies (incl. gaps): {int(df.label.sum()):,}")
