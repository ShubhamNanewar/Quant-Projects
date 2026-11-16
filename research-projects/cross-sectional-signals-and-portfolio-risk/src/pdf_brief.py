"""Render a shareable four-page brief from the checked-in research figures."""
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
ROOT=Path(__file__).resolve().parents[1]
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Deck',fontName='Helvetica',fontSize=12,leading=17,textColor=colors.HexColor('#486075'),spaceAfter=14))
styles['Title'].fontSize=25;styles['Title'].leading=29;styles['Title'].alignment=0
styles['BodyText'].fontSize=10;styles['BodyText'].leading=14;styles['BodyText'].spaceAfter=9
styles['Heading2'].textColor=colors.HexColor('#176a75')
flow=[]
def p(t,style='BodyText'):flow.append(Paragraph(t,styles[style]))
def chart(name,width=480):
    path=ROOT/'figures'/name;w,h=ImageReader(str(path)).getSize()
    flow.append(Image(str(path),width=width,height=width*h/w));flow.append(Spacer(1,8))
def page(n,title):
    if n>1:flow.append(PageBreak())
    p(f'RESEARCH BRIEF / {n:02d}','Heading2');p(title,'Title')
def footer(c,d):
    c.setStrokeColor(colors.HexColor('#d0d8df'));c.line(48,42,547,42)
    c.setFont('Helvetica',8);c.setFillColor(colors.HexColor('#526474'))
    c.drawString(48,28,'Shubham Vijay Nanewar | Research-proxy study | November 2026')
    c.drawRightString(547,28,str(d.page))
page(1,'Cross-Sectional Signals<br/>and Portfolio Risk')
p('Does a statistical return model improve on momentum when portfolio construction and costs are treated consistently?','Deck')
p('The simpler benchmark performed better. Across 180 evaluation months, constrained momentum achieved a 0.60 zero-hurdle Sharpe versus 0.30 for constrained Ridge. This study reports that negative model result rather than selecting only a favourable specification.')
chart('01_performance.png')
p('Scope and interpretation','Heading2')
p('48 US industry return aggregates; evaluation 2010-2024. Base charges: 10bp per unit of absolute traded weight and 50bp annually on short exposure. Returns are hypothetical. The aggregates are not directly tradable, and no capacity or live performance claim is made.')
p('The contribution is an auditable research pipeline: explicit timing, simple benchmarks, constrained optimisation, holdings-based accounting and uncertainty analysis. The full report and source code accompany this brief.')
page(2,'From observation to portfolio')
chart('00_workflow.png')
p('Prediction and validation','Heading2')
p('Six features describe reversal, two momentum horizons, volatility, drawdown and beta. Current cross-sectional ranks avoid full-sample scaling. Expanding Ridge predicts next-month industry returns relative to the cross-sectional average. Initial estimation covers 1991-2004; 2005-2009 validation selects one of four regularisation penalties. Evaluation covers 2010-2024; all four portfolio variants are retained.')
p('What is known at each decision?','Heading2')
p('Features and covariance stop at the penultimate trading day of the preceding month. Only labels completed before that cutoff enter training, excluding the incomplete preceding month. Hypothetical execution occurs at the next trading close. This prevents algorithmic look-ahead conditional on the inputs; it does not establish real-time publication availability of French aggregates.')
p('Risk and implementation','Heading2')
p('The optimiser uses Ledoit-Wolf covariance, zero net and estimated proxy-beta exposure, a 100% gross ceiling, a 5% per-industry cap and an L1 turnover penalty. These constraints apply at rebalancing. Signed holdings then drift with prices; the simulator deducts trading costs and daily short charges. Initial entry is charged; terminal positions are marked without liquidation.')
p('The optimiser uses a fixed rank-score utility scale, not calibrated expected returns. Comparing constrained and simple rank portfolios changes the risk mandate, so that comparison alone cannot establish signal superiority.')
page(3,'Costs and statistical uncertainty')
chart('03_costs.png')
p('Implementation matters','Heading2')
p('Annual absolute turnover was 4.99 for constrained momentum and 7.84 for constrained Ridge. At 50bp transaction costs, with short charges unchanged, their CAGRs were approximately 0.76% and -1.78%. Cost stresses replay fixed targets; they do not retune the strategy for each cost assumption.')
chart('07_uncertainty.png')
p('Paired circular 12-month block bootstrap, 2,000 repetitions. The 95% interval for the constrained Ridge-minus-momentum Sharpe difference is approximately [-0.62, -0.04]. This is a conditional retrospective diagnostic, not proof of future performance. Block length, nonstationarity and specification choices limit inference.')
page(4,'What the result does and does not show')
p('Research conclusion','Heading2')
p('In this specification, combining six features with Ridge did not improve the ranking of future returns enough to beat momentum. Mean rank IC was 0.025 for Ridge and 0.035 for momentum. Higher Ridge turnover also increased cost drag. A more complex model is therefore not justified by these results.')
p('Evidence quality','Heading2')
p('The repository includes predictions, monthly and daily returns, weights, covariance-risk diagnostics, timing audits, coefficients and cost stresses. Ten focused tests cover future-data isolation, label availability, optimisation constraints and hand-checkable holdings accounting. Source and protocol hashes identify the exact inputs.')
p('Important limits','Heading2')
p('The inherited Kenneth French CSV contains two return blocks. The second is explicitly labelled equal-weighted; the first is interpreted as value-weighted from provider ordering, but its original heading and download metadata are missing. A fresh download was unavailable. The snapshot is revised historical data, not a point-in-time archive. These provenance limits prevent a production-grade claim.')
p('Industry aggregates omit constituent turnover, borrow availability, realistic market impact and capacity. Costs are illustrative. There is no cash yield or short rebate. The protocol was written before this run, but this is a retrospective temporal evaluation, not an untouched prospective experiment.')
p('Next experiment','Heading2')
p('Before changing the model, obtain a verified data vintage and an investable universe with point-in-time membership. Then specify a new evaluation protocol, separate signal ablations from risk constraints, and test stability across economically distinct periods. Preserve the current negative finding as the baseline.')
p('Sources and reproduction','Heading2')
p('<link href="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_48_ind_port.html" color="#176a75">Kenneth French: 48 industry portfolio construction</link><br/><link href="https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html" color="#176a75">Kenneth French Data Library: methodology and revisions</link><br/><link href="https://github.com/ShubhamNanewar/Quant-Projects/tree/main/research-projects/cross-sectional-signals-and-portfolio-risk" color="#176a75">Code, full report, protocol and reproducible outputs</link>')
p('AI-assisted implementation was used. This project should be reproduced and understood before being presented as evidence of personal research proficiency. No affiliation with an investment firm is implied.')
SimpleDocTemplate(str(ROOT/'Research_Brief.pdf'),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=42,bottomMargin=55,title='Cross-Sectional Signals and Portfolio Risk',author='Shubham Vijay Nanewar').build(flow,onFirstPage=footer,onLaterPages=footer)
