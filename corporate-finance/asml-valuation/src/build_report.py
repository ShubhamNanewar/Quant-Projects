"""Produce the research note, a six-page brief and reusable vector figures."""
from pathlib import Path
import json
import html
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics import renderSVG

ROOT=Path(__file__).resolve().parents[1]
DATA=json.loads((ROOT/'data/processed/model_data.json').read_text())
REPORTS=ROOT/'reports'
INK=colors.HexColor('#17324F')
TEAL=colors.HexColor('#167B8A')
GOLD=colors.HexColor('#C58F28')
SOURCES=[
 ('ASML 2025 Form 20-F, US GAAP','https://www.sec.gov/Archives/edgar/data/937966/000162828026011378/asml-20251231.htm'),
 ('ASML 2024 Form 20-F, US GAAP','https://www.sec.gov/Archives/edgar/data/937966/000093796625000009/asml-20241231.htm'),
 ('ASML Q3 2026 statements','https://www.sec.gov/Archives/edgar/data/937966/000162828026065813/financial_statementsxusx.htm'),
 ('ASML October 2026 outlook','https://www.investor.asml.com/news-releases/news-release-details/asml-reports-eu11.7-billion-total-net-sales-and-eu5.0-billion-net-income-q3-2026'),
 ('ASML 2024 Investor Day','https://investor.asml.com/news-releases/news-release-details/asml-provides-updated-view-market-opportunities-2024-investor-day-meeting'),
 ('ASML products','https://www.asml.com/en/products'),
 ('Peer financials','https://data.sec.gov/api/xbrl/companyfacts/CIK0000006951.json'),
 ('Dated market prices','https://query1.finance.yahoo.com/v8/finance/chart/ASML.AS?period1=1793664000&period2=1793836800&interval=1d'),
]


def number(x):
 return f'{x:,.1f}'


def bar_chart():
 drawing=Drawing(490,220)
 chart=VerticalBarChart();chart.x=45;chart.y=38;chart.width=420;chart.height=160
 chart.data=[[DATA['historical'][str(y)][k]/1000 for y in [2023,2024,2025]] for k in ['Revenue','EBIT','Reported FCF']]
 chart.categoryAxis.categoryNames=['2023','2024','2025'];chart.valueAxis.valueMin=0;chart.valueAxis.valueMax=35;chart.valueAxis.valueStep=5
 chart.bars[0].fillColor=INK;chart.bars[1].fillColor=TEAL;chart.bars[2].fillColor=GOLD
 chart.groupSpacing=14;chart.barSpacing=2;drawing.add(chart)
 drawing.add(String(45,208,'Reported performance | EUR billion',fontName='Helvetica',fontSize=10,fillColor=INK))
 for i,(label,color) in enumerate([('Revenue',INK),('EBIT',TEAL),('Reported FCF',GOLD)]):
  drawing.add(String(65+i*140,10,label,fontName='Helvetica',fontSize=10,fillColor=color))
 return drawing


def report_markdown():
 a=DATA['historical']['2025'];base=DATA['results']['Base'];cases=DATA['results']
 rows='\n'.join(f'| {name} | {r["forecast"][-1]["revenue"]/1000:.1f} | {r["per_share"]:.2f} | {r["terminal_share"]:.1%} |' for name,r in cases.items())
 md=f'''# ASML: financial analysis and valuation

Independent project by Shubham Vijay Nanewar. Information cutoff and market reference: 4 November 2026. Prepared 5 November 2026. All ASML financial amounts are EUR million unless specified.

## Research question

What operating performance and cash conversion would support ASML's observed share price, and how much of a valuation depends on long-term assumptions?

This project combines reported financial analysis, a five-year operating forecast, a discounted cash flow model and a historical peer-multiple comparison. It is an independent learning exercise. It is not transaction experience, an investment recommendation, or a fully integrated three-statement model.

## Business and commercial context

ASML supplies lithography systems and related software, inspection and services used by semiconductor manufacturers. EUV supports advanced patterning, while DUV serves a broader set of manufacturing steps. Customer support and upgrades provide a second revenue source linked to the installed equipment base. The important economic questions concern customer capital expenditure, lithography intensity, equipment delivery and acceptance, capacity constraints and service demand.

Revenue recognition does not occur simply because an order is announced. Backlog, bookings and recognised revenue describe different stages of the commercial process. Customer advances help fund production but also create delivery obligations. They should not be treated as recurring profit or as unrestricted surplus capital.

ASML's 2024 Investor Day described potential 2030 revenue of EUR44-60 billion and gross margins of 56-60%. The October 2026 release subsequently guided 2026 revenue to EUR44-45 billion and discussed stronger customer demand. These are dated company statements, not guarantees. The base scenario reaches approximately EUR58.9 billion revenue in 2030. The upside scenario exceeds the older range and explicitly assumes a stronger demand and execution outcome.

## Reported financial performance

FY2025 revenue was {a['Revenue']/1000:.2f} billion, EBIT was {a['EBIT']/1000:.2f} billion and reported free cash flow was {a['Reported FCF']/1000:.2f} billion. Revenue increased 15.6% from 2024. EBIT margin was {a['EBIT']/a['Revenue']:.1%}; gross margin was {a['Gross profit']/a['Revenue']:.1%}. The analysis uses US GAAP consistently. IFRS capitalisation of development costs would change the profit comparison and is not mixed into the model.

Reported free cash flow is operating cash flow less cash purchases of property, plant and equipment and intangibles. It is not the same quantity as DCF unlevered free cash flow. The 2025 figure also reflects working-capital timing and customer funding. The simplified core working-capital measure, including current and noncurrent customer advances, was {a['Core NWC incl advances']/1000:.2f} billion. It excludes other operating assets and liabilities and is not a complete operating working-capital balance.

![Historical results](figures/historical_results.svg)

## Forecast design

The forecast starts from the EUR44.5 billion midpoint of the October 2026 revenue guidance. Forecast years are 2027-2031. Revenue growth, gross margin, R&D, SG&A, operating tax, depreciation, cash capex and normalised working capital are explicit inputs. Assumptions are editable in Excel; one case selector drives the same forecast and valuation.

Base revenue growth is 8%, 8%, 7%, 6% and 5%. Gross margin rises from 56% to 58%; R&D remains 10.5% of revenue and SG&A 3.2%. The combination implies operating leverage and an EBIT margin of 44.3% by 2031. This is a substantial improvement from FY2025 and deserves scrutiny. The downside holds gross margin at 53-54% with slower growth and higher expense ratios. The upside assumes stronger growth, 57.5-62% gross margin and greater operating leverage.

Normalised working capital is 10% of revenue, with a 2026 estimated starting balance of EUR4.45 billion. This is an analyst convention that deliberately does not extrapolate the reported negative balance funded by customer advances. It is not a forecast roll-forward of actual receivables, inventory and customer deposits. A model of each of those balances would be needed to judge how sustainable prepayment funding is.

## DCF method and results

Unlevered free cash flow equals EBIT after operating tax, plus depreciation, less cash capex and increases in normalised working capital. R&D remains expensed. Share-based compensation is not added back. Annual cash flows use year-end discounting. The remaining 2026 stub uses an explicit Q4 forecast, prorated for 57 remaining days out of a 95-day quarter. This assumes even cash generation within the quarter and is not a substitute for October actuals.

The discount rate is an illustrative 9.5%, not an empirically estimated WACC. Terminal growth is 3%. The terminal cash flow is rebuilt using the operating assumptions and the reinvestment required by terminal growth; it is not simply the last forecast cash flow multiplied by 1.03. The sensitivity table varies WACC from 7.5% to 11.5% and terminal growth from 2% to 4%.

| Scenario | 2031 revenue, EUR bn | DCF EUR/share | Terminal share of EV |
|---|---:|---:|---:|
{rows}

The observed 4 November close is EUR{DATA['market_price']:.2f}. The base model gives EUR{base['per_share']:.2f}, approximately {base['per_share']/DATA['market_price']-1:.1%} relative to that price. The gap does not establish mispricing. It shows that the market price requires more favourable operating assumptions, a lower discount rate, stronger long-term cash generation, or some combination. Terminal value represents {base['terminal_share']:.1%} of base enterprise value, so a precise-looking per-share result should not be mistaken for precise knowledge.

## Enterprise-to-equity adjustments

The bridge adds Q3 cash and short-term investments, deducts a EUR2 billion assumed operating-cash reserve and deducts estimated debt. Debt is approximated from December 2025 borrowings less nine-month debt and finance-lease repayments; it is not an exact Q3 balance and mixes debt and lease cash movements. The bridge adds Q3 equity investments, equity-method investments and loans at book value because their returns are not included in forecast EBIT. Their fair values, restrictions and strategic importance are not appraised separately. The denominator is Q3 basic weighted-average shares, 383.1 million, used as a proxy for ending shares. These adjustments are visible rather than described as exact market-value inputs.

## Peer comparison

Applied Materials, Lam Research and KLA provide equipment-industry context. They are not identical businesses: ASML focuses on lithography, Applied Materials has broader equipment exposure, Lam focuses on deposition and etch, and KLA focuses on process control. Differences in growth, margins, service mix and technology position can justify different multiples.

The peer table combines 4 November 2026 prices with FY2025 net income, revenue, cash, debt and shares. Later split-adjusted share counts are used where restated in SEC data, but subsequent buybacks are not reconstructed. These are explicitly historical-denominator diagnostics, not current TTM or consensus forward trading comparables. No multiple-derived target price is presented. Ratios are calculated in each company's reporting currency, so no FX conversion is required. Absolute EUR and USD figures should not be added or compared without conversion. Some peer cash inputs exclude short-term investments, making EV/sales approximate.

## Risks and questions for a client discussion

The key operating risks are customer capital-expenditure delays, export restrictions, supplier and production constraints, equipment acceptance timing, new-product adoption and service demand. The model translates these into growth, margin and reinvestment assumptions; it does not attempt to forecast geopolitical events.

For a financing or strategic discussion, I would first ask whether the issue is valuation, funding capacity, investment timing or shareholder returns. The model can support those questions, but it cannot determine debt capacity, recommend an acquisition or establish a hedge without additional work on contractual cash flows, financing terms and client objectives.

## What I would improve next

Replace the stub cash-flow approximation with actual intra-quarter cash information, use an exact debt and diluted-share bridge, forecast customer advances and inventory separately, estimate a market-supported discount rate, and rebuild peer metrics on matched TTM periods and dated share counts. These improvements could materially change the valuation.

## Reproduction and checks

`src/analyse.py` parses consolidated XBRL facts, excludes dimensional disclosures, chooses the most precise repeated facts and performs a separate Python valuation. Six tests cover reported statement identities, source controls, cash-flow bridges, scenario ordering and sensitivity direction. `src/build_workbook.mjs` checks all three Excel cases and all 25 native sensitivity-table values against Python, changes a late-year growth input to confirm recalculation, restores the base case and scans for formula errors. Excel desktop interaction has not been tested.

## Sources

'''
 md+='\n'.join(f'- [{label}]({url})' for label,url in SOURCES)
 md+='\n- Peer SEC company facts: Lam CIK0000707549 and KLA CIK0000319201. Exact filing accession numbers are in the peer CSV and workbook.\n'
 (REPORTS/'RESEARCH_REPORT.md').write_text(md)
 return md


def pdf_brief():
 styles=getSampleStyleSheet()
 styles.add(ParagraphStyle(name='BodyCustom',fontName='Helvetica',fontSize=10.5,leading=15,spaceAfter=10,textColor=INK))
 styles.add(ParagraphStyle(name='TitleCustom',fontName='Helvetica-Bold',fontSize=21,leading=25,spaceAfter=16,textColor=INK))
 styles.add(ParagraphStyle(name='SmallCustom',fontName='Helvetica',fontSize=8,leading=11,spaceAfter=6,textColor=INK))
 story=[]
 def para(text,small=False):story.append(Paragraph(text,styles['SmallCustom' if small else 'BodyCustom']))
 def heading(text):story.append(Paragraph(text,styles['TitleCustom']))
 def table(rows,widths=None):
  data=[[Paragraph(str(x),styles['SmallCustom']) for x in row] for row in rows]
  t=Table(data,colWidths=widths or [490/len(rows[0])]*len(rows[0]),hAlign='LEFT')
  t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF0F6')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.7,INK),('LINEBELOW',(0,-1),(-1,-1),.4,colors.HexColor('#B5C3D1'))]));story.append(t);story.append(Spacer(1,12))
 def nextpage():story.append(PageBreak())
 base=DATA['results']['Base'];a=DATA['historical']['2025']
 heading('ASML: business, cash flow and valuation')
 para('Independent financial-analysis project | Shubham Vijay Nanewar<br/>Market reference: 4 November 2026 | Prepared: 5 November 2026')
 para('The question is what operating performance would support ASML\'s share price. The approach connects reported financials to operating assumptions, free cash flow and a valuation range.')
 table([['Case','DCF EUR/share','2031 sales, EUR bn'],*[[n,f'{v["per_share"]:,.2f}',f'{v["forecast"][-1]["revenue"]/1000:.1f}'] for n,v in DATA['results'].items()]])
 para(f'Observed close: EUR{DATA["market_price"]:,.2f}. Base terminal value represents {base["terminal_share"]:.1%} of enterprise value. These figures are model outputs, not an investment recommendation.')
 para('ASML supplies lithography equipment, software and related services to chipmakers. EUV, DUV and installed-base support have different demand and margin drivers. Equipment orders, shipment, acceptance and revenue recognition are distinct events.')
 para('Commercial questions: How quickly can customers absorb additional capacity? How durable are upgrades and service demand? How do delivery timing, export restrictions and prepayments affect cash generation?')
 para('Sources: ASML products page, 2025 US GAAP annual report and October 2026 release. Full links are in the workbook and research report.',True)
 nextpage();heading('Reported performance and cash conversion')
 story.append(bar_chart());story.append(Spacer(1,8))
 table([['EUR billion','2023','2024','2025'],*[ [k,*[f'{DATA["historical"][str(y)][k]/1000:.2f}' for y in [2023,2024,2025]]] for k in ['Revenue','EBIT','Net income','Reported FCF']]])
 para(f'FY2025 gross margin was {a["Gross profit"]/a["Revenue"]:.1%}; EBIT margin was {a["EBIT"]/a["Revenue"]:.1%}. Reported FCF is operating cash flow less cash PPE and intangible purchases. It is not DCF unlevered cash flow.')
 para('Customer advances help fund equipment production. Core working capital, including current and noncurrent advances, was negative EUR8.0 billion in 2025. That funding benefit creates future delivery obligations and should not be extrapolated without examining its sustainability.')
 para('Source: SEC-filed ASML 2025 and 2024 Form 20-F statements; US GAAP throughout.',True)
 nextpage();heading('Operating assumptions and forecast')
 para('The 2026 revenue anchor is EUR44.5 billion, the midpoint of company guidance issued in October. Forecasts for 2027-2031 are analyst assumptions. The base reaches EUR58.9 billion in 2030, within the older EUR44-60 billion Investor Day scenario range. The upside exceeds it.')
 table([['Base assumptions','2027','2031'],['Revenue growth','8.0%','5.0%'],['Gross margin','56.0%','58.0%'],['R&D / revenue','10.5%','10.5%'],['SG&A / revenue','3.2%','3.2%'],['Operating tax','18.0%','18.0%'],['Capex / revenue','4.5%','4.5%'],['Normalised NWC / revenue','10.0%','10.0%']])
 table([['Base, EUR bn',*[str(x['year']) for x in base['forecast']]],['Revenue',*[f'{x["revenue"]/1000:.1f}' for x in base['forecast']]],['EBIT',*[f'{x["ebit"]/1000:.1f}' for x in base['forecast']]],['FCFF',*[f'{x["fcff"]/1000:.1f}' for x in base['forecast']]]])
 para('The base implies meaningful operating leverage. Normalised working capital is a positive analyst assumption, not a roll-forward of reported customer advances. R&D stays expensed and share-based compensation is not added back.')
 para('Sources: ASML October 2026 release and 2024 Investor Day. All later operating drivers are assumptions.',True)
 nextpage();heading('DCF and sensitivity')
 para('FCFF = EBIT after operating tax + D&A - cash capex - increase in normalised NWC. The model includes a prorated remaining-2026 stub, five annual forecasts and a terminal cash flow rebuilt from operating and reinvestment assumptions.')
 table([['Base bridge','EUR billion'],['Enterprise value',f'{base["ev"]/1000:.2f}'],['Add Q3 cash and ST investments','12.74'],['Less assumed minimum operating cash','(2.00)'],['Less estimated debt','(2.70)'],['Add non-operating assets at book value','5.45'],['Equity value',f'{base["equity"]/1000:.2f}']])
 table([['WACC / growth','2.0%','2.5%','3.0%','3.5%','4.0%'],*[[f'{w:.1%}',*[f'{v:,.0f}' for v in DATA['sensitivity'][i]]] for i,w in enumerate([.075,.085,.095,.105,.115])]])
 para('The 9.5% discount rate is illustrative, not an estimated market WACC. Debt is approximated using 2025 debt less nine-month debt and finance-lease repayments. Q3 weighted-average basic shares proxy ending shares. The stub assumes even quarterly cash generation. These are material simplifications.')
 para('Sources: Q3 2026 statements and independent calculations. Sensitivity outputs are EUR/share.',True)
 nextpage();heading('Peer comparison and interpretation')
 rows=[['Company','FY end','P/E','EV / sales']]
 for p in DATA['peers']:
  cap=p['price']*p['shares'];rows.append([p['symbol'],p['fiscal_end'],f'{cap/p["income"]:.1f}x',f'{(cap+p["debt"]-p["cash"])/p["revenue"]:.1f}x'])
 table(rows,[110,130,120,130])
 para('These are 4 November 2026 prices divided by FY2025 earnings and revenue. FY2025 shares and balance sheets are also used. They are historical-denominator diagnostics, not current TTM or forward trading comparables. Subsequent buybacks are not captured; restated split-adjusted shares are used where available.')
 para('Applied Materials, Lam Research and KLA have different equipment, services and customer exposures. Their multiples provide context but cannot be applied mechanically to ASML. Ratios use each company\'s own currency; absolute EUR and USD amounts are not aggregated.')
 para('The market price exceeds this model\'s three operating cases at the common 9.5% discount rate. A lower discount rate, stronger demand, better margins or a different long-term cash-flow view could narrow the gap. This is a question to investigate, not proof that the shares are mispriced.')
 para('Next steps: construct matched TTM figures, use dated ending shares and complete cash balances, and assess differences in growth and business mix before forming a multiple-based valuation.')
 para('Sources: SEC company-facts files and dated Yahoo chart responses, retained with provenance.',True)
 nextpage();heading('Discussion points and model limits')
 para('<b>What drives the result?</b> Growth matters, but terminal margins and the discount-rate/terminal-growth gap dominate a large portion of the valuation. A high-quality business can still require demanding assumptions at a high market price.')
 para('<b>A finding to discuss:</b> Reported cash generation strengthened much more than profit across the historical period. The question worth explaining is how working-capital timing and customer advances contribute, rather than assuming cash conversion is a permanent operating improvement.')
 para('<b>What would I ask a client?</b> First establish whether the decision concerns valuation, funding, capex timing or capital returns. An acquisition recommendation or debt-capacity assessment would require additional work on contractual commitments, financing terms and cash-flow downside protection.')
 para('<b>What is incomplete?</b> This is not a full three-statement forecast. The discount rate, normalised NWC, operating cash reserve and stub are assumptions. Debt and shares are proxies; non-operating assets are carried at book. The peer table is not matched TTM. Full sources and replication instructions accompany the model.')
 para('<b>Checks performed:</b> statement identities, reported source controls, cash-flow identities, scenario ordering and valuation monotonicity. Excel cases and the 25-cell native sensitivity table are compared with an independent Python model. Excel desktop testing was not performed.')
 para('<b>Interview preparation:</b> explain why US GAAP matters; distinguish reported FCF from FCFF; walk from enterprise to equity value; defend the base growth and margin assumptions; discuss how advance payments affect valuation; change an Excel assumption and explain the result.')
 para('Supporting files: ASML_Valuation.xlsx; RESEARCH_REPORT.md; INTERVIEW_GUIDE.md; data/processed CSVs; source manifest; Python tests.',True)
 def footer(c,doc):
  c.setStrokeColor(colors.HexColor('#D2DBE4'));c.line(48,43,547,43);c.setFont('Helvetica',8);c.setFillColor(INK)
  c.drawString(48,30,'ASML | Independent financial analysis | 5 November 2026');c.drawRightString(547,30,str(doc.page))
 SimpleDocTemplate(str(REPORTS/'ASML_Investment_Brief.pdf'),pagesize=(595,842),leftMargin=50,rightMargin=50,topMargin=45,bottomMargin=55).build(story,onFirstPage=footer,onLaterPages=footer)


def html_report(md):
 # Keep the source text inspectable; no web service or third-party chart script is required.
 import re
 content=[]
 in_table=False
 for line in md.splitlines():
  if line.startswith('|'):
   if not in_table:content.append('<table>');in_table=True
   cells=[x.strip() for x in line.strip('|').split('|')]
   if all(re.fullmatch(r'[:\-]+',c) for c in cells):continue
   content.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>');continue
  if in_table:content.append('</table>');in_table=False
  if line.startswith('!['):content.append('<img src="figures/historical_results.svg" alt="Historical financial performance">');continue
  if line.startswith('# '):content.append('<h1>'+html.escape(line[2:])+'</h1>')
  elif line.startswith('## '):content.append('<h2>'+html.escape(line[3:])+'</h2>')
  elif line.strip():
   safe=html.escape(line)
   safe=re.sub(r'\[([^\]]+)\]\((https://[^)]+)\)',r'<a href="\2">\1</a>',safe)
   content.append('<p>'+safe+'</p>')
 if in_table:content.append('</table>')
 css='body{font:17px/1.65 Arial,sans-serif;color:#17324f;max-width:960px;margin:48px auto;padding:0 24px;background:#fafbfc}h1{font-size:36px;line-height:1.2}h2{margin-top:40px;border-bottom:1px solid #cbd5df;padding-bottom:8px}table{border-collapse:collapse;width:100%;background:white}td{padding:10px;border-bottom:1px solid #dce3ea}tr:first-child{font-weight:bold;background:#eaf0f6}img{max-width:100%;background:white}a{color:#126e85}'
 (REPORTS/'REPORT.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>ASML financial analysis</title><style>'+css+'</style><body>'+''.join(content)+'</body></html>')


def main():
 REPORTS.mkdir(exist_ok=True);(REPORTS/'figures').mkdir(exist_ok=True)
 renderSVG.drawToFile(bar_chart(),str(REPORTS/'figures/historical_results.svg'))
 md=report_markdown();html_report(md);pdf_brief()
 print('Research note, HTML report and six-page brief written')


if __name__=='__main__':main()
