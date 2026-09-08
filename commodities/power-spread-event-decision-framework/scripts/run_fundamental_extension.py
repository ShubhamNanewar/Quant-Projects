"""Run the frozen 2025 comparison after fetch_fundamental_extension.py."""
from pathlib import Path
import hashlib
import json
import os
import platform

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fundamental_extension import build_panel, fixed_year_predictions


def main():
    source = ROOT/'data/raw/fundamental_extension'
    inputs = [ROOT/'data/raw/caiso_th_np15_da_rt_2024.csv', source/'prices_2025.csv', source/'demand_forecasts.csv']
    prices = pd.concat([pd.read_csv(path) for path in inputs[:2]],ignore_index=True)
    panel = build_panel(prices, pd.read_csv(inputs[2]))
    predictions, metrics, coverage = fixed_year_predictions(panel)
    out = ROOT/'reports/tables/fundamental_extension'
    out.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(out/'predictions.csv',index=False)
    metrics.to_csv(out/'summary.csv',index=False)
    coverage.to_csv(out/'coverage.csv',index=False)
    monthly = predictions.groupby(['model','month']).apply(lambda g:pd.Series({
        'hours':len(g),'mae':(g.prediction-g.spread).abs().mean(),
        'selected_hours':g.selected.sum(),
        'selected_mean_signed_spread':g.loc[g.selected,'signed_spread'].mean(),
        'selected_mean_less_1':g.loc[g.selected,'signed_spread'].mean()-1
    }),include_groups=False)
    monthly.to_csv(out/'monthly.csv')
    # Paired daily resampling: forecast-error difference, not a trading return.
    errors = predictions.assign(error=(predictions.prediction-predictions.spread).abs()).pivot(index='datetime_utc',columns='model',values='error')
    differences = errors[['rf_price','hourly_baseline']].sub(errors.rf_price_demand,axis=0)
    daily = differences.groupby(differences.index.floor('D')).agg(['sum','count'])
    rng = np.random.default_rng(7)
    intervals = {}
    for model in ['rf_price','hourly_baseline']:
        samples=[]
        for _ in range(2000):
            draw=daily.iloc[rng.integers(0,len(daily),len(daily))][model]
            samples.append(draw['sum'].sum()/draw['count'].sum())
        intervals[model]={'mean_mae_improvement':float(differences[model].mean()),
                          'daily_bootstrap_95pct_interval':np.quantile(samples,[.025,.975]).tolist()}
    manifest={'python':platform.python_version(),'sklearn':sklearn.__version__,
              'source_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
              'comparison':intervals,'note':'Positive error improvement favours demand model. Daily bootstrap conditions on fitted models and assumes independent days.',
              'availability':'Archived DAM inputs; publication/revision vintages not available. Forecasting experiment, not executable trading.'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    fig, axes = plt.subplots(1,3,figsize=(13,4),layout='constrained')
    labels=['Hourly baseline','RF price','RF price + demand']
    for ax,col,title in zip(axes,['mae','directional_accuracy','selected_mean_less_1_usd_per_mwh'],
                            ['2025 MAE ($/MWh)','2025 directional accuracy','Signed-spread proxy less $1/MWh']):
        ax.bar(labels,metrics[col],color=['#627D98','#007F86','#D78532'])
        ax.set_title(title,fontsize=11)
        ax.tick_params(axis='x',labelrotation=25)
        ax.spines[['top','right']].set_visible(False)
        if col.startswith('selected'):ax.axhline(0,color='gray',linewidth=.8)
    fig.suptitle('2025 forecast test: frozen models and thresholds\nSigned-spread scenarios are not executable trading P&L',fontsize=12)
    figs=ROOT/'reports/figures/fundamental_extension';figs.mkdir(parents=True,exist_ok=True)
    fig.savefig(figs/'2025_comparison.png',dpi=150)
    plt.close(fig)
    print(metrics.to_string(index=False),flush=True)
    print(coverage.to_string(index=False),flush=True)
    print(json.dumps(intervals,indent=2),flush=True)

if __name__=='__main__':
    main()
