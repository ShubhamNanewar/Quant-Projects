import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from research import load_returns,build_panel,fit_predict,rank_weights,optimise,simulate_month,metrics

def test_loader_separates_return_blocks_and_converts_units():
    r=load_returns()
    assert r.index.is_unique and r.shape[1]==48
    assert r.index.max()==pd.Timestamp('2024-12-31')
    assert r.abs().max().max()<1

def test_future_mutation_cannot_change_earlier_features():
    r=load_returns().loc['1990':'1992'];a=build_panel(r)
    changed=r.copy();changed.loc['1992-07-01':]=.3;b=build_panel(changed)
    for x,y in zip(a,b):
        if x['signal_date']<pd.Timestamp('1992-07-01'):
            np.testing.assert_allclose(x['x'],y['x'])
            np.testing.assert_allclose(x['covariance'],y['covariance'])

def test_execution_is_after_signal_and_before_returns():
    for row in build_panel(load_returns().loc['1990':'1992']):
        assert row['signal_date']<row['execution_date']<row['start']

def test_incomplete_month_label_is_excluded():
    rng=np.random.default_rng(2);rec=[]
    dates=pd.date_range('1990-01-31',periods=125,freq='ME')
    for d in dates:
        rec.append(dict(end=d,signal_date=d-pd.Timedelta(days=33),x=rng.normal(size=(48,6)),target=rng.normal(size=48)))
    pred,_,end=fit_predict(rec,124,1)
    assert end<rec[124]['signal_date']
    rec[123]['target'][:]=1000000
    pred2,_,_=fit_predict(rec,124,1)
    np.testing.assert_allclose(pred,pred2)

def test_rank_portfolio_is_neutral_and_unit_gross():
    w=rank_weights(np.arange(48))
    assert abs(w.sum())<1e-12 and abs(np.abs(w).sum()-1)<1e-12
    assert np.abs(w).max()<=.05

def test_optimiser_constraints_and_zero_signal():
    rng=np.random.default_rng(3);n=48;beta=rng.uniform(.5,1.5,n)
    w,_,err=optimise(rng.normal(size=n),np.eye(n)*.01,beta,np.zeros(n))
    assert err<1e-6 and abs(w.sum())<1e-7 and abs(w@beta)<1e-7
    assert np.abs(w).sum()<=1+1e-7 and np.abs(w).max()<=.05+1e-7
    z,_,_=optimise(np.zeros(n),np.eye(n)*.01,beta,np.zeros(n))
    np.testing.assert_allclose(z,0,atol=1e-9)

def test_holdings_drift_and_cost_accounting_hand_example():
    d=pd.DataFrame([[.1,0],[0,.1]],index=pd.date_range('2020-01-01',periods=2))
    rows,drift,net,gross,_=simulate_month(np.array([.5,-.5]),np.zeros(2),d,10,0)
    assert abs(net+.001)<1e-12 and abs(gross)<1e-12
    np.testing.assert_allclose(drift,[.55/.999,-.55/.999])
    assert abs(np.prod(1+rows.net_return)-1-net)<1e-12

def test_flat_returns_charge_entry_and_borrow():
    d=pd.DataFrame(np.zeros((252,2)),index=pd.bdate_range('2020-01-01',periods=252))
    _,_,net,_,borrow=simulate_month(np.array([.5,-.5]),np.zeros(2),d,10,50)
    assert abs(borrow-.0025)<1e-12 and abs(net-(-.0035))<1e-10

def test_turnover_uses_drifted_weights():
    d=pd.DataFrame([[0.,0.]],index=[pd.Timestamp('2020-01-01')])
    rows,_,_,_,_=simulate_month(np.array([.5,-.5]),np.array([.6,-.4]),d,10,0)
    assert abs(rows.turnover.sum()-.2)<1e-12

def test_drawdown_includes_initial_loss():
    assert abs(metrics(np.array([-.1,0,0]))['max_drawdown']+.1)<1e-12
