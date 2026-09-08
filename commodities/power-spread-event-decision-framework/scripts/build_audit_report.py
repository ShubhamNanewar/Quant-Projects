"""Build figures and a reproducibility record from the saved validation audit."""
from pathlib import Path
import os, hashlib, json, platform
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
T=ROOT/'reports/tables/validation_audit'
F=ROOT/'reports/figures/validation_audit'
F.mkdir(parents=True,exist_ok=True)
raw_path=ROOT/'data/raw/caiso_th_np15_da_rt_2024.csv'
raw=pd.read_csv(raw_path)
p=pd.read_csv(T/'predictions.csv',parse_dates=['datetime_utc'])
s=pd.read_csv(T/'summary.csv').set_index('model')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150})
colors=['#627D98','#007F86','#D78532']
names=['Hourly baseline','Random Forest','Ridge']
fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for ax,col,title in zip(axs,['mae','directional_accuracy','mean_spread_selected'],['Forecast MAE ($/MWh)','Directional accuracy','Selected-hour mean spread ($/MWh)']):
 ax.bar(names,s.loc[['hourly_baseline','random_forest','ridge'],col],color=colors)
 ax.set_title(title); ax.tick_params(axis='x',labelrotation=20)
 if col=='mean_spread_selected': ax.axhline(1,color='#B33F40',ls='--',label='Illustrative $1/MWh hurdle'); ax.legend(fontsize=8)
fig.suptitle('Delayed-feature audit: a higher hit rate does not establish a tradable edge',fontsize=13)
fig.savefig(F/'01_model_comparison.png'); plt.close(fig)
rf=p[p.model=='random_forest'].copy(); rf['month']=rf.datetime_utc.dt.strftime('%Y-%m')
selected=rf[rf.selected]
month=selected.groupby('month').spread.agg(['mean','size'])
fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
axs[0].bar(month.index,month['mean'],color=['#007F86' if v>=0 else '#B33F40' for v in month['mean']]); axs[0].axhline(0,color='gray',lw=.8)
axs[0].set_ylim(min(0, month['mean'].min())-0.8, month['mean'].max()+0.8)
axs[0].set(title='Mean selected spread by UTC month',ylabel='$/MWh'); axs[0].tick_params(axis='x',labelrotation=35)
for i,(v,n) in enumerate(zip(month['mean'],month['size'])): axs[0].annotate(f'n={n}',(i,v),xytext=(0,4 if v>=0 else -13),textcoords='offset points',ha='center',fontsize=8)
axs[1].hist(selected.spread,bins=30,color='#007F86'); axs[1].axvline(0,color='gray',lw=.8)
axs[1].set(title='Selected hours still include adverse outcomes',xlabel='Realized RT minus DA spread ($/MWh)',ylabel='Hours')
fig.savefig(F/'02_selection_risk.png'); plt.close(fig)
# Price-component decomposition is ex post attribution, not proof of physical cause.
raw['datetime_utc']=pd.to_datetime(raw.datetime_utc,utc=True)
local=raw.datetime_utc.dt.tz_convert('America/Los_Angeles')
raw['hour']=local.dt.hour; raw['month']=local.dt.month
components=['energy_spread','congestion_spread','loss_spread']
residual=raw.spread-raw[components].sum(axis=1)
# Published price components are rounded; tolerance is $0.0001/MWh.
if residual.abs().max()>1e-4: raise ValueError('Component identity failed')
events=raw.loc[raw.spread.abs().nlargest(6).index].sort_values('datetime_utc')
events[['datetime_utc','spread',*components]].to_csv(T/'largest_absolute_events.csv',index=False)
fig,axs=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
events.set_index(events.datetime_utc.dt.strftime('%m-%d %H:%M'))[components].plot.bar(ax=axs[0],color=['#007F86','#D78532','#627D98'])
axs[0].set(title='Six largest absolute spreads: price components',xlabel='Delivery hour (UTC)',ylabel='$/MWh'); axs[0].legend(['Energy','Congestion','Loss'],fontsize=8)
heat=raw.pivot_table(index='month',columns='hour',values='spread',aggfunc='mean')
limit=np.abs(heat.to_numpy()).max(); im=axs[1].imshow(heat,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
axs[1].set(title='Full-year mean spread: descriptive, not a signal',xlabel='Local delivery hour (Pacific)',ylabel='Month',yticks=np.arange(12),yticklabels=np.arange(1,13),xticks=[0,6,12,18,23])
fig.colorbar(im,ax=axs[1],label='$/MWh')
fig.savefig(F/'03_market_context.png'); plt.close(fig)
# Resample complete UTC delivery days; selections and models remain fixed.
rf['day']=rf.datetime_utc.dt.floor('D')
daily=rf.assign(value=rf.spread.where(rf.selected,0),count=rf.selected.astype(int)).groupby('day')[['value','count']].sum()
rng=np.random.default_rng(7); samples=[]
for _ in range(2000):
 b=daily.iloc[rng.integers(0,len(daily),len(daily))]
 if b['count'].sum(): samples.append(b.value.sum()/b['count'].sum())
ci=np.quantile(samples,[.025,.975])
meta={'input_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),'raw_rows':len(raw),
      'raw_start_utc':str(raw.datetime_utc.min()),'raw_end_utc':str(raw.datetime_utc.max()),
      'out_of_sample_hours':len(rf),'test_start_utc':str(rf.datetime_utc.min()),'test_end_utc':str(rf.datetime_utc.max()),
      'folds':int(rf.fold.nunique()),'max_component_identity_error':float(residual.abs().max()),
      'rf_selected_mean_daily_bootstrap_95pct_interval':ci.tolist(),
      'bootstrap_caveat':'Fixed predictions; IID resampled UTC days; excludes model-selection and multi-day dependence uncertainty.',
      'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__,
      'configuration':{'feature_lag_hours':48,'training_min_hours':4320,'calibration_hours':720,'embargo_hours':48,'test_hours':720,'signal_quantile':.9}}
(T/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
