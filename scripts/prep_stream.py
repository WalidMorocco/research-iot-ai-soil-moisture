import os, glob, pandas as pd
from dotenv import load_dotenv

load_dotenv()
RAW_DIR   = os.getenv("RAW_DIR")
CLEAN_DIR = os.getenv("CLEAN_DIR")
os.makedirs(CLEAN_DIR, exist_ok=True)

def _to_ts(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(dict(year   =df.year,
                             month  =df.month,
                             day    =df.day,
                             hour   =df.hour,
                             minute =df.minute,
                             second =df.second))
    return df.drop(columns=["year","month","day","hour","minute","second"])\
             .set_index(ts)

# 60-second files (plant_vase1 + duplicate)
files60 = [f for f in glob.glob(f"{RAW_DIR}/plant_vase1*.CSV")
           if "(2)" not in f]
df = pd.concat([_to_ts(pd.read_csv(f)) for f in files60]).sort_index()

long = (df.reset_index()
          .melt(id_vars=["index"],
                value_vars=[c for c in df.columns if c.startswith("moisture")],
                var_name="sensor_id",
                value_name="moisture")
          .rename(columns={"index":"ts"})
          .sort_values(["sensor_id","ts"]))

# gap > 120 s  → anomaly flag
long["gap_flag"] = (long.groupby("sensor_id")["ts"]
                         .diff().dt.total_seconds()
                         .gt(120)
                         .fillna(False))

out_path = os.path.join(CLEAN_DIR, "soil_stream_native.csv")
long.to_csv(out_path, index=False)

print(f"Saved {len(long):,} rows  (gap rows: {int(long.gap_flag.sum()):,})")
print(long.sensor_id.value_counts().to_string())
