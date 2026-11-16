"""Retrospective industry signal research; research portfolios are not tradable instruments."""
from __future__ import annotations
import hashlib, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'capital-shortfall-under-geopolitical-stress/data/48_Industry_Portfolios_Daily.csv'
FEATURES = ['reversal_21', 'momentum_63', 'momentum_252_21', 'volatility_63', 'drawdown_252', 'beta_252']
ALPHAS = [0.1, 1., 10., 100.]
VARIANTS = ['Momentum rank', 'Ridge rank', 'Momentum optimised', 'Ridge optimised']


def load_returns(path=SOURCE):
    raw = pd.read_csv(path, low_memory=False, encoding='utf-8-sig')
    key = raw.columns[0]
    boundary = raw[key].astype(str).str.contains('Average Equal Weighted', regex=False)
    if boundary.sum() != 1:
        raise ValueError('Expected exactly one labelled equal-weight block boundary')
    block = raw.iloc[:np.flatnonzero(boundary)[0]]
    valid = block[key].astype(str).str.fullmatch(r'\d{8}')
    block = block.loc[valid].copy()
    dates = pd.to_datetime(block.pop(key), format='%Y%m%d')
    block.index = dates
    block = block.astype(float).loc['1990-01-01':'2024-12-31'] / 100
    if block.index.duplicated().any() or not block.index.is_monotonic_increasing:
        raise ValueError('Duplicate or unordered dates')
    if block.shape[1] != 48 or not np.isfinite(block.to_numpy()).all() or (block <= -.999).any().any():
        raise ValueError('Invalid industry panel or missing-return sentinel')
    return block


def rank_scores(frame):
    """Ranks across the current cross-section only, scaled to approximately [-1.7,1.7]."""
    return (frame.rank(pct=True) - frame.rank(pct=True).mean()) * np.sqrt(12)


def features_at(history):
    if len(history) < 252:
        raise ValueError('252 observations needed')
    h = history.iloc[-252:]
    market = h.mean(axis=1)
    cov = h.cov().to_numpy()
    beta = cov.mean(axis=1) / market.var(ddof=1)
    wealth = (1+h).cumprod()
    # Include wealth of 1 at start so first-day losses enter drawdown.
    peak = wealth.cummax().clip(lower=1)
    raw = pd.DataFrame({
        'reversal_21': -np.expm1(np.log1p(h.iloc[-21:]).sum()),
        'momentum_63': np.expm1(np.log1p(h.iloc[-63:]).sum()),
        'momentum_252_21': np.expm1(np.log1p(h.iloc[:-21]).sum()),
        'volatility_63': h.iloc[-63:].std(ddof=1)*np.sqrt(252),
        'drawdown_252': (wealth/peak-1).min(),
        'beta_252': beta,
    }, index=h.columns)
    return rank_scores(raw), beta, LedoitWolf().fit(h.to_numpy()).covariance_ * 21


def build_panel(returns):
    months = list(returns.groupby(returns.index.to_period('M')))
    records = []
    for k in range(1, len(months)):
        month, future = months[k]
        previous = months[k-1][1]
        cutoff = previous.index[-2]
        history = returns.loc[:cutoff]
        if len(history) < 252:
            continue
        x, beta, covariance = features_at(history)
        target = np.expm1(np.log1p(future).sum()).to_numpy()
        records.append(dict(month=str(month), start=future.index[0], end=future.index[-1],
                            signal_date=cutoff, execution_date=previous.index[-1],
                            x=x.to_numpy(), beta=beta, covariance=covariance,
                            target=target-target.mean(), asset_returns=target,
                            future=future))
    return records


def fit_predict(records, current, alpha):
    row = records[current]
    # Strict inequality excludes the incompletely observed previous month.
    train = [r for r in records[:current] if r['end'] < row['signal_date']]
    if len(train) < 120:
        raise ValueError('At least 120 completed training months required')
    x = np.vstack([r['x'] for r in train]); y = np.concatenate([r['target'] for r in train])
    # Normalised mean-square objective makes penalties comparable as sample grows.
    model = Ridge(alpha=alpha*len(y), fit_intercept=False).fit(x, y)
    return model.predict(row['x']), model.coef_, train[-1]['end']


def spearman(a, b):
    ar=pd.Series(a).rank().to_numpy(); br=pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ar,br)[0,1])


def rank_weights(signal):
    scores = rank_scores(pd.Series(signal)).to_numpy()
    gross = np.abs(scores).sum()
    return scores/gross if gross else scores


def optimise(signal, covariance, beta, previous, penalty=.001, risk_aversion=5.):
    """Convex QP with epigraph variables for gross exposure and L1 turnover."""
    n=len(signal); eye=np.eye(n); zero=np.zeros((n,n))
    mu = rank_scores(pd.Series(signal)).to_numpy() * .01
    # variables: w, absolute w, absolute (w-previous)
    rows=np.vstack([np.hstack([eye,-eye,zero]), np.hstack([-eye,-eye,zero]),
                    np.hstack([eye,zero,-eye]), np.hstack([-eye,zero,-eye]),
                    np.r_[np.zeros(n),np.ones(n),np.zeros(n)][None,:]])
    upper=np.r_[np.zeros(2*n),previous,-previous,1.]
    equal=np.vstack([np.r_[np.ones(n),np.zeros(2*n)],np.r_[beta,np.zeros(2*n)]])
    constraints=[LinearConstraint(rows,-np.inf,upper),LinearConstraint(equal,0,0)]
    def fun(z):
        w=z[:n]
        return risk_aversion*.5*w@covariance@w-mu@w+penalty*z[2*n:].sum()
    def jac(z):
        return np.r_[risk_aversion*covariance@z[:n]-mu,np.zeros(n),np.full(n,penalty)]
    init=np.r_[np.zeros(2*n),np.abs(previous)]
    result=minimize(fun,init,jac=jac,method='SLSQP',constraints=constraints,
                    bounds=Bounds(np.r_[np.full(n,-.05),np.zeros(2*n)],
                                  np.r_[np.full(2*n,.05),np.full(n,5.)]),
                    options={'ftol':1e-10,'maxiter':300})
    w=result.x[:n]
    violation=max(abs(w.sum()),abs(w@beta),max(0,np.abs(w).sum()-1),max(0,np.abs(w).max()-.05))
    if not result.success or violation>1e-6:
        raise RuntimeError(f'Optimiser failure: {result.message}, violation={violation}')
    return w, result.nit, violation


def simulate_month(target, previous, daily, cost_bps=10, borrow_bps=50):
    """Self-financing holdings, cash with zero interest; costs at preceding execution close.

    Target weights are fractions of pre-cost NAV. Deduct trading costs first, then let
    dollar positions drift without daily rebalancing. Return drifted weights for next trade.
    """
    turnover=float(np.abs(target-previous).sum())
    cost=turnover*cost_bps/1e4
    nav=1-cost; holdings=target.copy(); start_nav=1.
    gross_holdings=target.copy(); gross_nav=1.
    rows=[]; total_borrow=0
    for t,(date,r) in enumerate(daily.iterrows()):
        borrow=np.maximum(-holdings,0).sum()*borrow_bps/1e4/252
        pnl=holdings@r.to_numpy(); g_pnl=gross_holdings@r.to_numpy()
        new_nav=nav+pnl-borrow; new_gross=gross_nav+g_pnl
        # First day includes prior-close initial cost in its recorded return.
        denom=start_nav if t==0 else nav
        rows.append(dict(date=date,net_return=new_nav/denom-1,
                         gross_return=new_gross/gross_nav-1,
                         gross_exposure=np.abs(holdings).sum()/nav,
                         net_exposure=holdings.sum()/nav,
                         turnover=turnover if t==0 else 0,borrow_cost=borrow,
                         trading_cost=cost if t==0 else 0))
        holdings*=1+r.to_numpy(); gross_holdings*=1+r.to_numpy()
        nav=new_nav; gross_nav=new_gross; total_borrow+=borrow
        if nav<=0: raise RuntimeError('Bankruptcy')
    return pd.DataFrame(rows),holdings/nav,nav-1,gross_nav-1,total_borrow


def metrics(r):
    r=np.asarray(r); wealth=np.cumprod(1+r); peak=np.maximum.accumulate(np.r_[1.,wealth])[1:]
    vol=r.std(ddof=1)*np.sqrt(12)
    return dict(cagr=wealth[-1]**(12/len(r))-1,volatility=vol,
                sharpe_zero_hurdle=r.mean()*12/vol if vol else 0,
                max_drawdown=float((wealth/peak-1).min()),total_return=wealth[-1]-1)


def block_bootstrap_difference(a,b,seed=42,reps=2000,block=12):
    rng=np.random.default_rng(seed);n=len(a);result=[]
    for _ in range(reps):
        starts=rng.integers(0,n,size=int(np.ceil(n/block)))
        ix=np.concatenate([(s+np.arange(block))%n for s in starts])[:n]
        result.append(metrics(a[ix])['sharpe_zero_hurdle']-metrics(b[ix])['sharpe_zero_hurdle'])
    return np.quantile(result,[.025,.5,.975])


def run():
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    returns=load_returns(); records=build_panel(returns)
    selected=[]; cached={};coeffs=[];timing=[]
    for alpha in ALPHAS:
        ics=[]
        for i,row in enumerate(records):
            if not '2005-01'<=row['month']<='2009-12': continue
            pred,coef,end=fit_predict(records,i,alpha)
            cached[(row['month'],alpha)]=pred
            ics.append(spearman(pred,row['target']))
        selected.append(dict(alpha=alpha,mean_ic=np.mean(ics),months=len(ics)))
    validation=pd.DataFrame(selected); validation.to_csv(out/'validation.csv',index=False)
    best=float(validation.sort_values(['mean_ic','alpha'],ascending=[False,True]).iloc[0].alpha)
    print('Validation selected alpha',best,flush=True)
    signals=[]; main=[];days=[];weights=[];diag=[];cost_results=[]
    variants={v:np.zeros(48) for v in VARIANTS}
    testrows=[]
    for i,row in enumerate(records):
        if not '2010-01'<=row['month']<='2024-12':continue
        pred,coef,end=fit_predict(records,i,best)
        momentum=row['x'][:,FEATURES.index('momentum_252_21')]
        timing.append(dict(month=row['month'],signal_date=row['signal_date'],execution_date=row['execution_date'],last_training_label=end,first_return=row['start']))
        coeffs.append(dict(month=row['month'],**dict(zip(FEATURES,coef))))
        signals.extend([dict(month=row['month'],asset=asset,prediction=pred[j],target=row['target'][j],momentum=momentum[j]) for j,asset in enumerate(returns.columns)])
        targetweights={}
        for v in VARIANTS:
            signal=pred if v.startswith('Ridge') else momentum
            if 'optimised' in v:
                w,nit,err=optimise(signal,row['covariance'],row['beta'],variants[v])
            else:w=rank_weights(signal);nit=0;err=0.
            targetweights[v]=w
            daily,drift,net,gross,borrow=simulate_month(w,variants[v],row['future'])
            variants[v]=drift
            main.append(dict(month=row['month'],strategy=v,net_return=net,gross_return=gross,
                             turnover=daily.turnover.sum(),borrow_cost=borrow,ic=spearman(signal,row['target'])))
            daily['strategy']=v;days.append(daily)
            diag.append(dict(month=row['month'],strategy=v,gross=np.abs(w).sum(),net=w.sum(),estimated_beta=w@row['beta'],predicted_volatility=np.sqrt(w@row['covariance']@w*12),max_weight=np.abs(w).max(),solver_iterations=nit,constraint_error=err))
            weights.extend([dict(month=row['month'],strategy=v,asset=asset,weight=w[j]) for j,asset in enumerate(returns.columns)])
        testrows.append((row,targetweights))
        if len(testrows)%36==0:print('Completed',len(testrows),'evaluation months',flush=True)
    monthly=pd.DataFrame(main); daily=pd.concat(days,ignore_index=True)
    summary=[]
    for v,g in monthly.groupby('strategy',sort=False):
        m=metrics(g.net_return); d=daily[daily.strategy==v].net_return.to_numpy();wealth=np.cumprod(1+d)
        m['daily_max_drawdown']=float((wealth/np.maximum.accumulate(np.r_[1,wealth])[1:]-1).min())
        summary.append(dict(strategy=v,**m,annual_turnover=g.turnover.mean()*12,mean_ic=g.ic.mean()))
    summary=pd.DataFrame(summary)
    # Replay fixed targets under alternative cost/borrow assumptions, retaining drift accounting.
    for cost,borrow in [(0,50),(5,50),(10,50),(25,50),(50,50),(0,0),(10,0),(10,100),(10,300)]:
        for v in VARIANTS:
            prev=np.zeros(48);rr=[]
            for row,ws in testrows:
                _,prev,net,_,_=simulate_month(ws[v],prev,row['future'],cost,borrow);rr.append(net)
            cost_results.append(dict(strategy=v,cost_bps=cost,borrow_bps=borrow,**metrics(rr)))
    intervals=[];pivot=monthly.pivot(index='month',columns='strategy',values='net_return')
    for a,b in [('Ridge rank','Momentum rank'),('Ridge optimised','Momentum optimised'),('Ridge optimised','Ridge rank')]:
        lo,med,hi=block_bootstrap_difference(pivot[a].to_numpy(),pivot[b].to_numpy())
        intervals.append(dict(comparison=a+' minus '+b,lower_95=lo,median=med,upper_95=hi))
    tables={'monthly_returns':monthly,'daily_returns':daily,'performance':summary,'weights':pd.DataFrame(weights),'risk_diagnostics':pd.DataFrame(diag),'predictions':pd.DataFrame(signals),'coefficients':pd.DataFrame(coeffs),'timing_audit':pd.DataFrame(timing),'cost_sensitivity':pd.DataFrame(cost_results),'bootstrap_intervals':pd.DataFrame(intervals)}
    for name,df in tables.items():df.to_csv(out/(name+'.csv'),index=False)
    manifest=dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),source_relative_path=str(SOURCE.relative_to(ROOT.parents[1])),source_vintage='Unknown inherited snapshot, last date 2025-11-28',study_start=str(returns.index[0].date()),study_end=str(returns.index[-1].date()),daily_rows=len(returns),industries=48,evaluation_months=len(testrows),alpha=best,protocol_sha256=hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),python=platform.python_version(),seed=42)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(summary.to_string(index=False),flush=True)
    print(pd.DataFrame(intervals).to_string(index=False),flush=True)

if __name__=='__main__':run()
