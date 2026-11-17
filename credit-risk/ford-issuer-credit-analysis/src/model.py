"""Reproducible historical issuer credit case. No live prices or probability claims."""
import csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
INPUT=json.loads((ROOT/'data/inputs.json').read_text())
def forecast(s,opening=None,maturity=None):
 o=INPUT['opening'] if opening is None else opening
 mat=INPUT['maturities']['2026'] if maturity is None else maturity
 fcf=s['ebit']+s['depreciation']-s['capex']+s['working_capital']+s['credit_distributions']-s['cash_interest']-s['cash_tax']+s['other_timing']
 post=fcf-s['restructuring']-s['pension']-s['dividends']-s['other_nonoperating']
 cash=o['cash_and_securities']+post-mat
 # Conservative full-year 8% interest on any liquidity-support draw: solve simultaneously.
 draw=max(0,(o['cash_target']-cash)/(1-.08))
 actual_draw=min(draw,max(0,o['durable_undrawn_facilities']))
 funded_cash=cash+actual_draw*(1-.08)
 remaining=max(0,o['durable_undrawn_facilities']-actual_draw)
 return dict(adjusted_fcf=fcf,cash_after_nonoperating=post,cash_before_funding=cash,headroom_to_target=cash-o['cash_target'],required_draw=draw,actual_draw=actual_draw,draw_interest=actual_draw*.08,cash_after_funding=funded_cash,remaining_facilities=remaining,unfunded_need=max(0,draw-actual_draw),ebit_interest_coverage=s['ebit']/s['cash_interest'])
def bond_price(y,coupon=.0325,years=6,face=100):
 n=int(years*2)
 return sum(face*coupon/2/(1+y/2)**t for t in range(1,n+1))+face/(1+y/2)**n

def write_csv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def run():
 rows=[dict(scenario=k,**forecast(v)) for k,v in INPUT['scenarios'].items()]
 write_csv(ROOT/'data/scenario_results.csv',rows)
 sensitivity=[]
 for e in [-2,0,2,4,6,8]:
  for wc in [-4,-2,0,2]:
   s=INPUT['scenarios']['Base'].copy();s.update(ebit=e,working_capital=wc)
   sensitivity.append(dict(ebit=e,working_capital=wc,**forecast(s)))
 write_csv(ROOT/'data/sensitivity.csv',sensitivity)
 price=bond_price(.06)
 write_csv(ROOT/'data/bond_sensitivity.csv',[dict(illustrative_yield=y,price_per_100=bond_price(y),change_vs_6pct=bond_price(y)/price-1) for y in [.04,.05,.06,.07,.08,.10]])
 # Exact published GAAP-to-adjusted cash-flow bridge, USD billions (p80).
 bridge=21.282-12.931-(-.720)-(-.436)-0-(-.996)-8.694+1.650+.054
 assert abs(bridge-3.513)<1e-9
 assert abs(sum(INPUT['maturities'].values())-INPUT['opening']['debt_including_leases'])<1e-9
 for r in rows:print(r)
 print('2025 GAAP-to-adjusted FCF reconciliation:',round(bridge,3))
 return rows
if __name__=='__main__':run()
