"""Run from the project directory: PYTHONPATH=src python scripts/run_desk_diagnostics.py"""
from pathlib import Path
import hashlib
import json
import pandas as pd
from desk_diagnostics import validate, summarize, cost_sensitivity

root=Path(__file__).resolve().parents[1]
source=root/"reports/tables/fundamental_extension/predictions.csv"
out=root/"reports/tables/desk_diagnostics"
out.mkdir(parents=True,exist_ok=True)
data=validate(pd.read_csv(source))
for label, keys in [("overall",["model"]),("by_delivery_hour",["model","delivery_hour"]),("by_season",["model","season"])]:
    summarize(data,keys).to_csv(out/(label+".csv"),index=False)
cost_sensitivity(data).to_csv(out/"cost_sensitivity.csv",index=False)
data.loc[data.selected].sort_values("signed_spread").groupby("model").head(10).to_csv(out/"worst_selected_hours.csv",index=False)
(out/"manifest.json").write_text(json.dumps({"input_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"rows":len(data),"purpose":"Post-hoc descriptive diagnostics; frozen models unchanged; not a trading backtest","timezone":"America/Los_Angeles"},indent=2)+"\n")
print(summarize(data,["model"]).to_string(index=False))
