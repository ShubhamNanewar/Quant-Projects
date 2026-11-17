"""Generate a six-page credit memo and standalone chart files. Requires ReportLab."""
from pathlib import Path
import json,sys,csv
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.graphics.shapes import Drawing,Rect,String,Line
from reportlab.graphics import renderSVG
from model import ROOT,INPUT,forecast,bond_price
S=getSampleStyleSheet();S['Title'].alignment=0;S['Title'].fontSize=25;S['Title'].leading=29
S['BodyText'].fontSize=10;S['BodyText'].leading=14;S['BodyText'].spaceAfter=8
S['Heading2'].fontSize=13;S['Heading2'].textColor=colors.HexColor('#096979');S['Heading2'].spaceBefore=10
S.add(ParagraphStyle(name='Note',fontSize=8,leading=11,textColor=colors.HexColor('#536373'),spaceAfter=6))
flow=[];navy=colors.HexColor('#14374d');teal=colors.HexColor('#00887e');orange=colors.HexColor('#d57931')
def p(t,style='BodyText'):flow.append(Paragraph(t,S[style]))
def page(n,title,subtitle):
 if n>1:flow.append(PageBreak())
 p(f'ISSUER CREDIT RESEARCH / {n:02d}','Heading2');p(title,'Title');p(subtitle,'Note');flow.append(Spacer(1,7))
def table(headers,rows,widths=None):
 data=[[Paragraph(str(x),S['Note']) for x in row] for row in [headers]+rows]
 t=Table(data,colWidths=widths or [499/len(headers)]*len(headers),hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e3eff1')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,teal),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dde2e6')),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));flow.append(t);flow.append(Spacer(1,9))
def bars(name,labels,values,title,target=None):
 d=Drawing(499,185);d.add(String(0,171,title,fontName='Helvetica-Bold',fontSize=11,fillColor=navy));lo=min(0,min(values));hi=max(max(values),target or 0)*1.15
 def yy(v):return 32+(v-lo)/(hi-lo)*120
 d.add(Line(35,yy(0),490,yy(0),strokeColor=colors.grey))
 w=420/len(values)
 for i,(label,v) in enumerate(zip(labels,values)):
  x=48+i*w;d.add(Rect(x,min(yy(0),yy(v)),w*.55,abs(yy(v)-yy(0)),fillColor=teal if v>=0 else orange,strokeColor=None));d.add(String(x,yy(v)+(5 if v>=0 else -12),f'{v:.2f}',fontSize=9));d.add(String(x,12,label,fontSize=8))
 if target is not None:
  d.add(Line(35,yy(target),490,yy(target),strokeColor=orange,strokeDashArray=[3,3]));d.add(String(350,yy(target)+4,'Management target: 20',fontSize=8,fillColor=orange))
 flow.append(d);renderSVG.drawToFile(d,str(ROOT/'figures'/f'{name}.svg'))
def footer(c,d):
 c.setStrokeColor(colors.HexColor('#ccd5da'));c.line(48,43,547,43);c.setFont('Helvetica',8);c.drawString(48,29,'Shubham Vijay Nanewar | Historical FY2025 case | Prepared 17 November 2026');c.drawRightString(547,29,str(d.page))
rows={k:forecast(v) for k,v in INPUT['scenarios'].items()}
page(1,'Ford: cash resilience<br/>versus refinancing risk','Industrial issuer focus: Ford Motor Company excluding Ford Credit. USD billions unless specified.')
p('Credit conclusion','Heading2')
p('The model supports a distinction between near-term repayment capacity and longer-term credit risk. Under the base assumptions, Ford can repay the scheduled 2026 principal and retain cash above its management target without new borrowing. In the downside and severe cases, preserving that target requires facility draws. Liquidity is therefore a buffer, not evidence that earnings deterioration is harmless.')
table(['Case','Adjusted FCF','Cash before funding','Draw to retain $20bn'],[[k,f"{v['adjusted_fcf']:.2f}",f"{v['cash_before_funding']:.2f}",f"{v['required_draw']:.2f}"] for k,v in rows.items()])
p('Investment stance: conditional, not a trade recommendation','Heading2')
p('The research hypothesis is that near-dated industrial debt may be more defensible than long-duration debt if available spreads compensate for issuer risk. A recommendation to buy requires executable prices, a matched benchmark curve, issue-level terms and a comparison with alternative issuers. None is assumed here. The model cannot establish that a bond is cheap or that a spread will tighten.')
p('Three questions driving the analysis','Heading2')
p('1. How much industrial cash generation remains after restructuring, pensions and distributions?<br/>2. Can scheduled debt be paid if bond-market access is unavailable?<br/>3. Would a credit deterioration damage long-duration bond prices even if immediate repayment remains feasible?')
p('Information boundary','Heading2')
p('This is a retrospective case built from the 2025 annual report and its original 2026 outlook. Subsequent 2026 results, financing changes and market prices are deliberately excluded. It is not an up-to-date assessment as of the preparation date. Scenario outputs are conditional calculations, not estimated probabilities.')
p('Primary source: Ford 2025 Annual Report, printed pp62-66, 74, 79-82 and Note18 pp156-159. Page references below use printed pages; PDF page = printed page + 6.','Note')
page(2,'Start with the correct cash flow','Historical diagnosis and accounting reconciliation')
bars('historical_fcf',['2023','2024','2025'],[6.801,6.672,3.513],'Company adjusted free cash flow ($bn)')
p('Avoid the consolidated cash-flow trap','Heading2')
p('Ford Credit funds receivables and leases, while the industrial business manufactures vehicles. Subtracting consolidated debt from industrial cash, or treating consolidated operating cash flow as industrial repayment capacity, mixes different economic perimeters. The model uses the company-excluding-Ford-Credit view and treats distributions from Ford Credit as a separate, stressable inflow.')
table(['2025 reconciliation to adjusted FCF','Cash-flow effect'],[['Consolidated operating cash flow','21.282'],['Remove Ford Credit operating cash flow','-12.931'],['Add back funded pension contributions / restructuring','+0.720 / +0.436'],['Other adjustment; tax sharing','+0.996; 0.000'],['Industrial capital spending','-8.694'],['Ford Credit distributions; derivative settlements','+1.650; +0.054'],['Company adjusted FCF','3.513']],[380,119])
p('Source: p80. These are reconciliation signs, not a statement that restructuring or pensions are economically free. Forecasts deduct those outflows below adjusted FCF.','Note')
p('What changed?','Heading2')
p('Rounded industrial adjusted EBIT fell from 8.6 to 4.2 between 2024 and 2025, while capital spending remained high. Credit distributions supported reported adjusted FCF. The underwriting question is whether cash conversion can recover without relying on distributions or timing benefits that may weaken during stress.')
p('Debt/EBITDA is not the lead metric','Heading2')
p('The rounded industrial EBIT plus depreciation proxy is 9.4; gross debt including leases divided by this proxy is about 2.33x. This is not a covenant ratio or rating-agency adjusted leverage. Lease, pension, cash-access and non-recurring-item adjustments would be needed for a formal comparison.')
page(3,'Debt ladder and funding access','Contractual obligations are distinct from accounting carrying values')
m=INPUT['maturities'];bars('maturities',list(m)[:-1],list(m.values())[:-1],'Industrial contractual principal by year ($bn)')
p('Principal schedule reconciles to carrying debt','Heading2')
p('Contractual principal totals 22.205; carrying adjustments of -0.286 reconcile it to 21.919. The schedule includes finance leases, so lease principal is not subtracted again in the scenario cash bridge. Ford Credit debt of 141.417 is outside this industrial bridge and cannot simply be ignored in an assessment of group contagion.')
table(['Funding or constraint','Treatment in the model'],[['13.5 corporate + 2.0 supplemental facilities','15.5 capacity before stress draws; assumed accessible, not guaranteed cash'],['2.5 short facility + 3.0 delayed-draw facility','Excluded from durable capacity because availability ends during 2026 under the disclosed terms'],['Opening cash and marketable securities','28.433 from p82; excludes separately classified restricted cash; cross-border remittance is not verified'],['20 management cash target','Analytical buffer threshold, not a default or covenant trigger'],['4 minimum liquidity covenant','Different domestic-cash/eligible-facility definition; no numerical covenant-compliance claim']],[170,329])
p('Sources: p62, p66, p82 and Note18 pp156-159. Facility assumptions exclude local lines. They assume no draw by another eligible borrower and no new restriction on availability.','Note')
p('Questions for legal and treasury diligence','Heading2')
p('Check issuer and guarantor entities, permitted liens, restricted subsidiaries, cross-default wording, change-of-control terms, and ranking for the particular bond. The annual report says certain subsidiary guarantees are required if two investment-grade ratings are not maintained. That is not equivalent to automatic acceleration. Issue-level documents remain necessary.')
page(4,'Stress the cash bridge','One-year 2026 scenarios: analyst assumptions, not management forecasts')
keys=[('Industrial adjusted EBIT','ebit'),('Depreciation','depreciation'),('Capital spending (outflow)','capex'),('Working-capital cash effect','working_capital'),('Ford Credit distributions','credit_distributions'),('Cash interest / cash tax',None),('Other and timing','other_timing'),('Restructuring / pensions',False),('Shareholder distributions','dividends')]
data=[]
for title,key in keys:
 vals=[]
 for s in INPUT['scenarios'].values():
  vals.append(f"{s['cash_interest']:.2f} / {s['cash_tax']:.2f}" if key is None else f"{s['restructuring']:.2f} / {s['pension']:.2f}" if key is False else f'{s[key]:.2f}')
 data.append([title]+vals)
table(['Driver','Base','Downside','Severe'],data,[235,88,88,88])
p('All three cases also deduct 0.30 other non-operating cash and 5.553 scheduled principal. No new bond issuance or asset-sale proceeds are assumed. Base FCF of 5.5 matches the midpoint of the original guidance by construction; this is not independent forecast validation. The 4.2 other/timing assumption is material and explicitly challenged below.','Note')
bars('cash_stress',list(rows),[v['cash_before_funding'] for v in rows.values()],'Cash after maturities, before new funding ($bn)',20)
p('Funding consequence','Heading2')
p('Draws are solved including an assumed full-year 8% cost of new borrowing: draw = max(0, (20 - pre-funding cash) / 0.92). This deliberately conservative timing convention produces 3.52 downside and 7.33 severe draws, leaving 11.98 and 8.17 of modelled facilities. Positive headroom does not prove intrayear liquidity or legal access.')
page(5,'What would change the conclusion?','Sensitivity, management actions and an event-monitoring framework')
s=INPUT['scenarios']['Downside'].copy();s['dividends']=0;cut=forecast(s)
s=INPUT['scenarios']['Base'].copy();s['other_timing']=0;timing=forecast(s)
table(['Independent sensitivity','Model implication'],[['Suspend downside distributions',f"Pre-funding cash rises to {cut['cash_before_funding']:.2f}; draw falls to {cut['required_draw']:.2f}. Discretionary action, not assured."],['Remove base other/timing benefit',f"Pre-funding cash falls to {timing['cash_before_funding']:.2f}; draw becomes {timing['required_draw']:.2f}. Base resilience depends on cash conversion."],['Additional 1.0 cash outflow','Cash falls by 1.0. Once below target, funding need rises by 1/0.92 = 1.09 under the assumed draw cost.'],['Facility availability falls below 7.33','Severe case cannot preserve the 20 target without another action; this is not itself a payment default.']],[190,309])
p('Working capital can move against the cycle','Heading2')
p('A production decline can reduce cash receipts while supplier payments still fall due. That is why the severe case combines lower EBIT with a working-capital outflow instead of assuming inventories automatically release cash. Annual totals cannot identify the worst intrayear cash point; a monthly model is the next diligence step.')
table(['Monitor','Credit interpretation / response'],[['Industrial earnings and price incentives','Check whether margin pressure is temporary or sustained; revise EBIT before drawing a spread conclusion.'],['Warranty cash payments and restructuring','Reconcile expense recognition with cash timing; avoid double-counting charges already in operating assumptions.'],['Ford Credit losses, residual values and funding','Test whether distributions fall or parent support is needed; the latter is not included in the current stress.'],['Ratings, facility renewal and bond issuance','Reassess access and guarantees. Separate a change in capacity from a change in funding cost.'],['Acquisitions, asset sales and other corporate actions','Map cash consideration, acquired debt, disposal proceeds and lost earnings before judging creditor impact.']],[190,309])
p('A news-response template','Heading2')
p('State the event and source time; identify the issuer and affected claims; quantify the cash/earnings channel; distinguish temporary timing from permanent impairment; update scenarios; then compare the revised view with observable bond prices. A confident narrative without a price comparison is not yet a trade idea.')
page(6,'Connect issuer risk to bond risk','Illustrative valuation mechanics, with no market quote or target price')
p('A six-year, 3.25% semiannual bond','Heading2')
p('Ford disclosed 3.25% notes due February 2032. The example below uses that coupon with an assumed six-year remaining term, settlement on a coupon date and principal of 100. It is an illustrative fixed-cash-flow calculation, not a valuation of the actual security: exact settlement, accrued interest and optionality are omitted.')
yields=[.04,.05,.06,.07,.08,.10];base=bond_price(.06)
table(['Assumed yield','Price per 100','Change vs 6% yield'],[[f'{y:.0%}',f'{bond_price(y):.2f}',f'{bond_price(y)/base-1:.1%}'] for y in yields])
p('Interpreting the sensitivity','Heading2')
p('A 200bp rise from 6% to 8% reduces the illustrative price even though all contractual payments are still assumed to occur. Yield includes both benchmark rates and credit/liquidity compensation; it is not a credit spread. This calculation is therefore a duration sensitivity, not a default-loss forecast. A default scenario would require separate recovery and timing assumptions.')
p('What is still needed before recommending a bond?','Heading2')
p('Obtain current bid/ask prices and timestamps, exact security identifiers and terms, a matched benchmark curve, spread duration and option-adjusted measures where appropriate. Compare with peers and nearby maturities on a consistent issuer/ranking basis. Then decide whether the observed compensation covers the downside, rather than inferring value from issuer solvency alone.')
p('Interview discussion','Heading2')
p('Be ready to explain why industrial and financing debt are separated; why adjusted FCF is not cash available to repay debt; how expiring facilities change the buffer; why the base case is anchored to guidance; and why a breach of the 20 cash target is not a covenant breach. Reproduce the model and change assumptions before presenting the project as your work.')
p('Reproduction and source','Heading2')
p('Run python src/model.py and python -m unittest discover -s src -p test_model.py. Regenerate this memo with python src/report.py after installing ReportLab. Inputs, assumptions, output tables and sensitivity grids are included. Implementation was AI-assisted; the model is an educational research case, not an audited forecast.')
p('<link href="https://s205.q4cdn.com/882619693/files/doc_financials/2025/ar/__Ford-2025-Annual-Report__.pdf" color="#096979">Primary source: Ford 2025 Annual Report</link>. Source PDF retained locally; source locations are recorded in data/inputs.json.','Note')
SimpleDocTemplate(str(ROOT/'Ford_Credit_Research.pdf'),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=40,bottomMargin=55,title='Ford: Cash Resilience versus Refinancing Risk',author='Shubham Vijay Nanewar').build(flow,onFirstPage=footer,onLaterPages=footer)
