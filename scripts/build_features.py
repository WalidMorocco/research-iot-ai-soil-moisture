import os, numpy as np, pandas as pd
from dotenv import load_dotenv
load_dotenv()

# 0. Paths
CLEAN_DIR = os.getenv("CLEAN_DIR")
IN_FILE   = os.path.join(CLEAN_DIR, "soil_stream_labeled.csv")
OUT_FILE  = os.path.join(CLEAN_DIR, "soil_stream_features.csv")

df = pd.read_csv(IN_FILE, parse_dates=["ts"])

# 1. Delta per sensor
df["delta"] = df.groupby("sensor_id")["moisture"].diff().fillna(0).round(2)

# 2. Hour sine/cos
df["hour_sin"] = np.sin(2*np.pi*df["ts"].dt.hour/24).round(2)
df["hour_cos"] = np.cos(2*np.pi*df["ts"].dt.hour/24).round(2)

# 3. Neighbour delta
pivot     = df.pivot(index="ts", columns="sensor_id", values="moisture")
nbr_mean  = pivot.mean(axis=1)
df["neighbor_delta"] = (pivot.sub(nbr_mean, axis=0)
                             .stack()
                             .reindex(df.set_index(["ts","sensor_id"]).index)
                             .values.round(2))

# 4. Robust z-scores
for col in ["moisture", "delta"]:
    q75 = df.groupby("sensor_id")[col].transform("quantile", 0.75)
    q25 = df.groupby("sensor_id")[col].transform("quantile", 0.25)
    df[f"{col}_z"] = ((df[col] - df.groupby("sensor_id")[col].transform("median"))
                      / (q75 - q25 + 1e-6)).round(2)

df.to_csv(OUT_FILE, index=False)
print("✓ 'label' column preserved." if "label" in df.columns else
      "⚠︎  'label' missing.")
print(f"Features saved → {OUT_FILE}")
