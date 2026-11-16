"""Generate figures and a standalone research report from saved results."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.mplconfig'))
import json,base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from matplotlib.patches import FancyBboxPatch

O=ROOT/'outputs';F=ROOT/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold','axes.titlesize':14,'savefig.dpi':160})
COLORS={'Momentum rank':'#a4acb7','Ridge rank':'#d28439','Momentum optimised':'#416b9a','Ridge optimised':'#14826e'}
monthly=pd.read_csv(O/'monthly_returns.csv');daily=pd.read_csv(O/'daily_returns.csv',parse_dates=['date'])
perf=pd.read_csv(O/'performance.csv');diag=pd.read_csv(O/'risk_diagnostics.csv');cost=pd.read_csv(O/'cost_sensitivity.csv')
coef=pd.read_csv(O/'coefficients.csv');boot=pd.read_csv(O/'bootstrap_intervals.csv');manifest=json.loads((O/'manifest.json').read_text())

def save(fig,name):
 fig.savefig(F/(name+'.png'),bbox_inches='tight');plt.close(fig)

fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
for v,c in COLORS.items():
 g=daily[daily.strategy==v];w=(1+g.net_return).cumprod();peak=np.maximum.accumulate(np.r_[1,w])[1:]
 axes[0].plot(g.date,w,label=v,color=c,lw=1.8);axes[1].plot(g.date,w/peak-1,color=c,lw=1.2)
axes[0].set(title='Prediction versus portfolio construction',ylabel='Hypothetical wealth (start = 1)');axes[0].legend(ncol=2,frameon=False)
axes[1].set(ylabel='Drawdown');axes[1].yaxis.set_major_formatter(PercentFormatter(1))
fig.suptitle('2010–2024 temporal evaluation | 10bp turnover cost + 50bp annual short charge',fontsize=10,y=.99)
fig.tight_layout();save(fig,'01_performance')

fig,ax=plt.subplots(figsize=(11,3.6))
years=monthly.assign(year=monthly.month.str[:4]).groupby(['strategy','year']).net_return.apply(lambda x:(1+x).prod()-1).unstack().reindex(COLORS)
mat=ax.imshow(years.to_numpy(),cmap='RdYlGn',vmin=-.15,vmax=.15,aspect='auto')
ax.set_xticks(range(len(years.columns)),years.columns,rotation=45);ax.set_yticks(range(4),years.index)
for i in range(4):
 for j in range(len(years.columns)): ax.text(j,i,f'{years.iloc[i,j]:.0%}',ha='center',va='center',fontsize=8)
ax.set_title('Annual net returns: no years omitted');fig.colorbar(mat,ax=ax,format=PercentFormatter(1),shrink=.7);fig.tight_layout();save(fig,'02_annual_returns')
years.to_csv(O/'annual_returns.csv')

fig,axes=plt.subplots(1,2,figsize=(11,4))
for v,c in COLORS.items():
 g=cost[(cost.strategy==v)&(cost.borrow_bps==50)].sort_values('cost_bps')
 axes[0].plot(g.cost_bps,g.cagr,'o-',label=v,color=c)
 g=cost[(cost.strategy==v)&(cost.cost_bps==10)].sort_values('borrow_bps')
 axes[1].plot(g.borrow_bps,g.cagr,'o-',color=c)
for ax in axes:ax.axhline(0,color='black',lw=.6);ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_ylabel('Annualised net return')
axes[0].set(title='Transaction-cost stress',xlabel='Basis points per unit traded');axes[1].set(title='Short-cost stress',xlabel='Annual basis points on short exposure')
axes[0].legend(fontsize=8,frameon=False);fig.tight_layout();save(fig,'03_costs')

fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
for v,c in COLORS.items():
 g=diag[diag.strategy==v];t=pd.to_datetime(g.month)
 axes[0].plot(t,g.estimated_beta,color=c,label=v,lw=1)
 axes[1].plot(t,g.predicted_volatility,color=c,lw=1)
axes[0].axhline(0,color='black',lw=.6);axes[0].set(title='Risk controls at each rebalance',ylabel='Estimated proxy beta');axes[0].legend(ncol=2,frameon=False)
axes[1].set(ylabel='Forecast annual volatility');axes[1].yaxis.set_major_formatter(PercentFormatter(1));fig.tight_layout();save(fig,'04_risk')

fig,ax=plt.subplots(figsize=(11,3.6))
for v in ['Momentum rank','Ridge rank']:
 g=monthly[monthly.strategy==v];ax.plot(pd.to_datetime(g.month),g.ic.rolling(12).mean(),label=v,color=COLORS[v])
ax.axhline(0,color='black',lw=.7);ax.set(title='Forecast quality changes over time',ylabel='12-month mean rank IC');ax.legend(frameon=False);fig.tight_layout();save(fig,'05_information_coefficient')

fig,ax=plt.subplots(figsize=(11,3.8))
a=coef.set_index('month');im=ax.imshow(a.T*100,cmap='RdBu_r',aspect='auto',vmin=-.15,vmax=.15)
ax.set_yticks(range(len(a.columns)),a.columns);ix=np.arange(0,len(a),24);ax.set_xticks(ix,a.index[ix],rotation=45)
ax.set_title('Expanding-window Ridge coefficients');fig.colorbar(im,ax=ax,label='Monthly return percentage points per rank-score unit');fig.tight_layout();save(fig,'06_coefficients')

fig,ax=plt.subplots(figsize=(10,3.8))
for i,r in boot.iterrows():
 ax.plot([r.lower_95,r.upper_95],[i,i],color='#14826e',lw=3)
 ax.scatter(r['median'],i,color='#143e52',s=45,zorder=3)
ax.axvline(0,color='#777',ls='--');ax.set_yticks(range(len(boot)),boot.comparison);ax.set(xlabel='Difference in annualised zero-hurdle Sharpe',title='Uncertainty: paired 12-month block bootstrap')
fig.tight_layout();save(fig,'07_uncertainty')

fig,ax=plt.subplots(figsize=(11,3.3));ax.axis('off')
labels=['48 industry\nreturn series','Lagged ranks\n+ completed labels','Expanding Ridge\nvs momentum','Risk-constrained\nportfolio weights','Drift, costs\n+ diagnostics']
for i,label in enumerate(labels):
 x=.01+i*.2;ax.add_patch(FancyBboxPatch((x,.43),.17,.4,boxstyle='round,pad=.01',facecolor='#e9f4f1',edgecolor='#14826e'))
 ax.text(x+.085,.63,label,ha='center',va='center',fontsize=10)
 if i<4:ax.annotate('',xy=(x+.195,.63),xytext=(x+.175,.63),arrowprops={'arrowstyle':'->','color':'#143e52'})
ax.text(.5,.14,'1991–2004 initial estimation   |   2005–2009 validation   |   2010–2024 temporal evaluation',ha='center',fontsize=11)
ax.set_title('Research design: from evidence to portfolio decisions');save(fig,'00_workflow')

# Portable HTML contains its own image assets. A human-readable Markdown version accompanies it.
def pct(x):return f'{x:.2%}'
def table_html(df):return df.to_html(index=False,border=0,float_format=lambda x:f'{x:.3f}')
rows=[]
for _,r in perf.iterrows():rows.append([r.strategy,pct(r.cagr),f'{r.sharpe_zero_hurdle:.2f}',pct(r.volatility),pct(r.daily_max_drawdown),f'{r.annual_turnover:.2f}',f'{r.mean_ic:.3f}'])
summary=pd.DataFrame(rows,columns=['Strategy','Net CAGR','Sharpe*','Volatility','Daily drawdown','Annual turnover','Rank IC'])
main=perf.set_index('strategy');delta=main.loc['Ridge optimised','sharpe_zero_hurdle']-main.loc['Momentum optimised','sharpe_zero_hurdle']
conclusion=f"The Ridge-optimised portfolio returned {main.loc['Ridge optimised','cagr']:.2%} annually with a zero-hurdle Sharpe of {main.loc['Ridge optimised','sharpe_zero_hurdle']:.2f}. Under identical portfolio constraints, momentum returned {main.loc['Momentum optimised','cagr']:.2%} with a Sharpe of {main.loc['Momentum optimised','sharpe_zero_hurdle']:.2f}. The observed Sharpe difference was {delta:.2f}; the bootstrap interval below, rather than that point estimate alone, determines how much confidence to place in it."
sections=[('Research question','Does combining return-history characteristics improve on a simple industry-momentum signal, and how much of any portfolio difference comes from construction rather than prediction? This study separates these two questions through a four-way comparison.'),('Design',f"Use 48 industry return proxies, with 1990 history for warm-up. Six features: short-term reversal, medium- and long-horizon momentum, volatility, drawdown and proxy beta. Cross-sectional ranks use only information through the penultimate trading day of the previous month. A Ridge model refits monthly on expanding completed labels. Validation selects alpha={manifest['alpha']:g} from four fixed choices. No evaluation-period tuning is performed."),('Result',conclusion),('How to interpret the portfolios','Rank portfolios have 100% gross exposure and zero net exposure at rebalance. The optimiser adds a 5% name cap, a 100% gross cap, estimated beta neutrality to the equal-industry proxy, covariance risk and an explicit turnover penalty. It uses the same score scale for both signals. These are different risk mandates: a higher optimised Sharpe cannot by itself prove a better forecasting model.'),('Trading-accounting assumptions','Positions are set at the previous month’s last close using earlier information. Holdings drift between monthly rebalances. Turnover compares target weights with drifted existing weights; costs are deducted before the new month’s returns. Base costs are 10bp per unit traded and 50bp per year on short exposure. Entry costs are included; terminal positions are marked, not liquidated. There is no cash yield, short rebate or positive financing income. Cost stresses hold target weights fixed.'),('Limitations that change the conclusion','The data are published research aggregates, not directly tradable securities. Exact constituent trades, borrow availability, spread, liquidity and capacity are absent. Source download metadata and the first-block heading are missing from the inherited CSV; its value-weighted interpretation follows the provider’s layout, not a fresh independent source check. Historical data can be revised; the snapshot is not a point-in-time archive. These results are not an executable strategy or evidence of scalable alpha. The temporal holdout was selected before this run, but is retrospective, not a prospective preregistration.'),('What would be needed next','Replicate with licensed point-in-time constituent data or a documented tradable ETF universe; include delistings and historical holdings; estimate instrument-level spreads, borrow and market impact; test against a genuine market factor and broader risk exposures; reserve genuinely unseen future data. Do not select another model on this evaluation period and continue calling it untouched.'),('Research integrity','All four variants and negative findings are retained. Ten tests check feature causality, training-label availability, optimiser constraints, holdings drift, entry and borrow charges, and turnover. Automated implementation assistance was used; all claims must be understood and independently reviewed by the author before presenting this as personal research expertise.')]
html=['<!doctype html><html><head><meta charset="utf-8"><title>Cross-sectional signals and portfolio risk</title><style>body{font:16px/1.65 system-ui;color:#20313c;background:#f4f7f8;margin:0}main{max-width:1120px;margin:auto;background:white;padding:48px}h1{font-size:38px;line-height:1.2}h2{margin-top:32px;color:#166d60}img{width:100%;height:auto;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}.sub{color:#60717c}.note{padding:16px;background:#e9f4f1}a{color:#166d60}</style></head><body><main><p class="sub">SYSTEMATIC INVESTMENT RESEARCH · Shubham Vijay Nanewar</p><h1>Cross-sectional signals &amp; portfolio risk</h1><p class="sub">48 industry portfolios · 180 evaluation months · Monthly walk-forward modelling</p>']
md=['# Cross-sectional signals and portfolio risk\n','Research study by Shubham Vijay Nanewar.\n']
for title,body in sections:
 html+=['<h2>'+title+'</h2>','<p>'+body+'</p>'];md+=['## '+title+'\n',body+'\n']
 if title=='Research question':names=['00_workflow']
 elif title=='Result':
  html+=[table_html(summary),'<p class="sub">*Monthly returns, annualised; zero hurdle, no cash yield. Research proxies, not executable returns.</p>'];md+=[summary.to_csv(index=False)];names=['01_performance','02_annual_returns','07_uncertainty']
 elif title=='Design':names=['05_information_coefficient','06_coefficients']
 elif title=='How to interpret the portfolios':names=['04_risk']
 elif title=='Trading-accounting assumptions':names=['03_costs']
 else:names=[]
 for n in names:
  html+=['<img alt="'+n+'" src="data:image/png;base64,'+base64.b64encode((F/(n+'.png')).read_bytes()).decode()+'">'];md+=['!['+n+'](figures/'+n+'.png)\n']
html+=['<h2>Sources and reproducibility</h2><p><a href="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_48_ind_port.html">Kenneth French industry construction</a> · <a href="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html">Data library and revision notes</a></p><p>Reproduce: python src/research.py; python src/present.py; pytest -q. See PROTOCOL.md, data/README.md and outputs/manifest.json for exact assumptions and source hash.</p></main></body></html>']
(ROOT/'RESEARCH_REPORT.html').write_text('\n'.join(html));(ROOT/'RESEARCH_REPORT.md').write_text('\n'.join(md))
summary.to_csv(O/'presentation_summary.csv',index=False)
print(conclusion)
