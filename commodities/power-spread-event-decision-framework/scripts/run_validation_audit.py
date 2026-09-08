from pathlib import Path
import pandas as pd
from validation_audit import audit_features, audit_walk_forward, diagnostic_summary
ROOT=Path(__file__).resolve().parents[1]
raw=pd.read_csv(ROOT/'data/raw/caiso_th_np15_da_rt_2024.csv')
f=audit_features(raw)
p,m=audit_walk_forward(f)
out=ROOT/'reports/tables/validation_audit'
out.mkdir(parents=True, exist_ok=True)
p.to_csv(out/'predictions.csv', index=False)
m.to_csv(out/'fold_metrics.csv', index=False)
s=diagnostic_summary(p)
s.to_csv(out/'summary.csv', index=False)
p['month']=p.datetime_utc.dt.strftime('%Y-%m')
monthly=p.groupby(['model','month']).apply(lambda g: pd.Series({
    'observations':len(g), 'selected_hours':int(g.selected.sum()),
    'selected_mean_spread':g.loc[g.selected,'spread'].mean(),
    'selected_positive_share':(g.loc[g.selected,'spread']>0).mean()
}), include_groups=False)
monthly.to_csv(out/'monthly_diagnostics.csv')
print(s.to_string(index=False))
