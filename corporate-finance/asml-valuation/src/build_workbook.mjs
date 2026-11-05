import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const data = JSON.parse(await fs.readFile(path.join(root,'data/processed/model_data.json'),'utf8'));
const wb = Workbook.create();
const names = ['Summary','DCF','Forecast','Assumptions','Actuals','Peers'];
const sheets = Object.fromEntries(names.map(n=>[n,wb.worksheets.add(n)]));
const ink='#172F4D', green='#00804A', blue='#174EA6';
const fmt='#,##0.0;(#,##0.0);"-"';
const cols=['F','G','H','I','J'];
function val(s,a,v){s.getRange(a).values=[[v]];}
function formula(s,a,v){s.getRange(a).formulas=[[v]];s.getRange(a).format.font.color=v.includes('!')?green:'#171717';}
function row(s,n,label,values){val(s,'C'+n,label);s.getRange(`F${n}:J${n}`).values=[values];}
function title(s,text,context){val(s,'C2',text);s.getRange('C2').format.font={size:16,bold:true,color:ink};val(s,'C3',context);}
function band(s,n,text){val(s,'C'+n,text);s.getRange(`C${n}:J${n}`).format={fill:ink,font:{bold:true,color:'#FFFFFF'}};}
for(const [name,s] of Object.entries(sheets)){
 s.showGridLines=false;s.getRange('A1:O80').format.font={name:'Arial',size:10,color:ink};
 s.getRange('A1:B80').format.columnWidth=2;s.getRange('C1:C80').format.columnWidth=42;
 s.getRange('D1:E80').format.columnWidth=14;s.getRange('F1:J80').format.columnWidth=15;
 s.getRange('K1:K80').format.columnWidth=3;s.getRange('L1:O80').format.columnWidth=17;
 s.getRange('C1:J80').format.rowHeight=21;s.getRange('D5:J80').setNumberFormat(fmt);
 s.tabColor=name==='Summary'?ink:name==='Assumptions'?'#597695':'#C8D3DF';
}
const ac=sheets.Actuals;
title(ac,'ASML | Reported financials','EUR million; shares in millions | US GAAP | audited annual figures');
const keys=Object.keys(data.historical['2025']);
const rows={};
val(ac,'C5','Reported metric');ac.getRange('F5:H5').values=[[2023,2024,2025]];
keys.forEach((key,i)=>{const r=i+6;rows[key]=r;val(ac,'C'+r,key);ac.getRange(`F${r}:H${r}`).values=[[2023,2024,2025].map(y=>data.historical[y][key])];});
ac.getRange('F6:H34').format.font.color=blue;
val(ac,'C37','2025 annual source: SEC Form 20-F');val(ac,'D38','https://www.sec.gov/Archives/edgar/data/937966/000162828026011378/asml-20251231.htm');
val(ac,'C40','2023 balance sheet source: 2024 Form 20-F');val(ac,'D41','https://www.sec.gov/Archives/edgar/data/937966/000093796625000009/asml-20241231.htm');
band(ac,43,'Q3 2026 | Reported figures and debt estimate inputs');
const snapshot=[['9M revenue',29755.4],['Q3 cash',8030.9],['Q3 short-term investments',4704.2],['2025 current debt',1681.9],['2025 long-term debt',2709],['9M debt/finance-lease repayments',1693.7],['Q3 equity investments',2239.4],['Q3 equity-method investments',1129.2],['Q3 current loans receivable',303],['Q3 noncurrent loans receivable',1782.5],['Q3 basic weighted shares (proxy)',383.1],['ASML close, 4 Nov 2026 (EUR)',data.market_price]];
snapshot.forEach(([k,v],i)=>{val(ac,'C'+(44+i),k);val(ac,'F'+(44+i),v);});
val(ac,'C58','Quarterly source (unaudited)');val(ac,'D59','https://www.sec.gov/Archives/edgar/data/937966/000162828026065813/financial_statementsxusx.htm');
val(ac,'C61','Prices: Yahoo chart API, daily unadjusted close, 4 November 2026. Raw JSON retained.');
val(ac,'C63','US GAAP is used throughout; IFRS R&D capitalisation is not mixed into these figures.');
ac.freezePanes.freezeRows(5);

const ass=sheets.Assumptions;
title(ass,'ASML | Forecast assumptions','Analyst assumptions, not company guidance except where explicitly identified');
val(ass,'C4','Case selector (1 / 2 / 3)');val(ass,'D4',1);
ass.getRange('D4').format={fill:'#FFF2CC',font:{color:blue,bold:true}};
ass.dataValidations.add({range:'D4',rule:{type:'whole',operator:'between',formula1:1,formula2:3}});
val(ass,'C5','Case selected');formula(ass,'D5','=CHOOSE(D4,"Base","Downside","Upside")');
ass.getRange('F5:J5').values=[[2027,2028,2029,2030,2031]];
const drivers=[['growth','Revenue growth'],['gross_margin','Gross margin'],['rd','R&D / sales'],['sga','SG&A / sales'],['tax','Operating tax rate'],['da','D&A / sales'],['capex','Cash capex / sales'],['nwc','Normalised NWC / sales']];
const active={};
drivers.forEach(([key,label],i)=>{
 const r=7+i*5;active[key]=r;val(ass,'C'+r,label+' | Active');
 ['Base','Downside','Upside'].forEach((n,j)=>row(ass,r+1+j,n,data.assumptions[key][j]));
 cols.forEach(c=>formula(ass,c+r,`=CHOOSE($D$4,${c}${r+1},${c}${r+2},${c}${r+3})`));
 ass.getRange(`C${r}:J${r}`).format.fill='#EAF0F6';ass.getRange(`F${r}:J${r+3}`).setNumberFormat('0.0%');
 ass.getRange(`F${r+1}:J${r+3}`).format.font.color=blue;
});
const other=[['FY2026 revenue midpoint (company Q3 guide)',44500],['FY2026 normalised NWC / revenue',.10],['Q4 2026 gross margin (company guide)',.57],['Q4 R&D (company guide)',1200],['Q4 SG&A (company guide)',400],['Q4 tax rate (assumption)',.17],['Q4 D&A (assumption)',330],['Q4 cash capex (assumption)',650],['Q4 increase in NWC (assumption)',500],['Minimum operating cash reserve',2000]];
other.forEach(([k,v],i)=>{val(ass,'C'+(49+i),k);val(ass,'F'+(49+i),v);});
for(const r of [50,51,54])ass.getRange('F'+r).setNumberFormat('0.0%');
val(ass,'C61','Normalised NWC excludes structural customer prepayment funding; it is not reported NWC.');
val(ass,'C47','Cases: 1 Base; 2 Downside; 3 Upside');
val(ass,'C63','Q4 guide source: ASML 14 October 2026 release; forecasts beyond 2026 are analyst scenarios.');
val(ass,'D64','https://www.investor.asml.com/news-releases/news-release-details/asml-reports-eu11.7-billion-total-net-sales-and-eu5.0-billion-net-income-q3-2026');
ass.freezePanes.freezeRows(5);

const fc=sheets.Forecast;
title(fc,'ASML | Operating forecast','EUR million | 2026 estimated anchor; annual forecasts 2027-2031');
val(fc,'C4','Case selected');formula(fc,'D4',"='Assumptions'!D5");
fc.getRange('E5:J5').values=[['2026E',2027,2028,2029,2030,2031]];
const labels={7:'Revenue',8:'Revenue growth',10:'Gross profit',11:'Gross margin',13:'Research and development',14:'SG&A',16:'EBIT',17:'EBIT margin',19:'Operating taxes',20:'NOPAT',22:'Depreciation and amortisation',23:'Cash capital expenditure',25:'Normalised working capital balance',26:'Change in normalised working capital',28:'Unlevered free cash flow'};
Object.entries(labels).forEach(([r,l])=>val(fc,'C'+r,l));
formula(fc,'E7',"='Assumptions'!F49");formula(fc,'E25',"=E7*'Assumptions'!F50");
cols.forEach((c,i)=>{
 const prev=i===0?'E':cols[i-1];
 for(const [r,key] of [[8,'growth'],[11,'gross_margin']])formula(fc,c+r,`='Assumptions'!${c}${active[key]}`);
 formula(fc,c+'7',`=${prev}7*(1+${c}8)`);formula(fc,c+'10',`=${c}7*${c}11`);
 formula(fc,c+'13',`=${c}7*'Assumptions'!${c}${active.rd}`);formula(fc,c+'14',`=${c}7*'Assumptions'!${c}${active.sga}`);
 formula(fc,c+'16',`=${c}10-SUM(${c}13:${c}14)`);formula(fc,c+'17',`=${c}16/${c}7`);
 formula(fc,c+'19',`=${c}16*'Assumptions'!${c}${active.tax}`);formula(fc,c+'20',`=${c}16-${c}19`);
 formula(fc,c+'22',`=${c}7*'Assumptions'!${c}${active.da}`);formula(fc,c+'23',`=${c}7*'Assumptions'!${c}${active.capex}`);
 formula(fc,c+'25',`=${c}7*'Assumptions'!${c}${active.nwc}`);formula(fc,c+'26',`=${c}25-${prev}25`);
 formula(fc,c+'28',`=SUM(${c}20,${c}22)-SUM(${c}23,${c}26)`);
});
for(const r of [8,11,17])fc.getRange(`E${r}:J${r}`).setNumberFormat('0.0%');
for(const r of [7,16,20,28])fc.getRange(`C${r}:J${r}`).format.borders={bottom:{style:'thin',color:'#A0ADBA'}};
val(fc,'C31','This is an operating forecast and DCF, not a fully integrated three-statement model.');
val(fc,'C33','Reported 2025 core NWC includes advances; the forecast uses positive normalised NWC.');
val(fc,'C35','R&D remains expensed; share-based compensation is not added back to cash flow.');

const d=sheets.DCF;
title(d,'ASML | Discounted cash flow','Reference date: 4 November 2026 | EUR million except per share');
val(d,'C4','Case selected');formula(d,'D4',"='Assumptions'!D5");
val(d,'C6','Discount rate (WACC assumption)');val(d,'D6',.095);
val(d,'C7','Terminal growth (assumption)');val(d,'D7',.03);
d.getRange('D6:D7').format={fill:'#FFF2CC',font:{color:blue}};d.getRange('D6:D7').setNumberFormat('0.0%');
val(d,'C9','Remaining days to 31 Dec 2026');val(d,'D9',57);val(d,'C10','Q4 period days (28 Sep - 31 Dec)');val(d,'D10',95);
val(d,'C12','Q4 revenue (guide midpoint less 9M actual)');formula(d,'D12',"='Assumptions'!F49-'Actuals'!F44");
val(d,'C13','Q4 EBIT');formula(d,'D13',"=D12*'Assumptions'!F51-SUM('Assumptions'!F52:F53)");
val(d,'C14','Q4 forecast FCFF');formula(d,'D14',"=D13*(1-'Assumptions'!F54)+'Assumptions'!F55-'Assumptions'!F56-'Assumptions'!F57");
val(d,'C15','Remaining Q4 FCFF (pro-rata estimate)');formula(d,'D15','=D14*D9/D10');
val(d,'C16','PV remaining Q4');formula(d,'D16','=D15/(1+D6)^(D9/365)');
d.getRange('F18:J18').values=[[2027,2028,2029,2030,2031]];
val(d,'C19','Forecast FCFF');val(d,'C20','Year-end discount exponent');val(d,'C21','Present value');
cols.forEach((c,i)=>{formula(d,c+'19',`='Forecast'!${c}28`);formula(d,c+'20',`=${i+1}+$D$9/365`);formula(d,c+'21',`=${c}19/(1+$D$6)^${c}20`);});
const detail={24:'Terminal-year revenue',25:'Terminal-year NOPAT',26:'Terminal-year net capex',27:'Terminal-year change in NWC',28:'Terminal FCFF',29:'Terminal enterprise value',30:'PV terminal value',32:'Enterprise value',33:'Cash and short-term investments',34:'Less minimum operating cash reserve',35:'Less estimated debt',36:'Non-operating investments and loans (book)',38:'Equity value',39:'Basic weighted shares (proxy)',40:'DCF value per share (EUR)',41:'Observed close, 4 Nov 2026 (EUR)',42:'Difference from observed close',43:'Terminal value / enterprise value'};
Object.entries(detail).forEach(([r,l])=>val(d,'C'+r,l));
formula(d,'D24',"='Forecast'!J7*(1+D7)");
formula(d,'D25',"=D24*('Assumptions'!J12-'Assumptions'!J17-'Assumptions'!J22)*(1-'Assumptions'!J27)");
// Terminal margins must follow the selected case, not the base-case input row.
formula(d,'D25',`=D24*('Assumptions'!J${active.gross_margin}-'Assumptions'!J${active.rd}-'Assumptions'!J${active.sga})*(1-'Assumptions'!J${active.tax})`);
formula(d,'D26',`=D24*('Assumptions'!J${active.capex}-'Assumptions'!J${active.da})`);
formula(d,'D27',"='Forecast'!J25*D7");formula(d,'D28','=D25-SUM(D26:D27)');
formula(d,'D29','=IF(D6>D7,D28/(D6-D7),NA())');formula(d,'D30','=D29/(1+D6)^J20');
formula(d,'D32','=SUM(F21:J21)+D16+D30');formula(d,'D33',"=SUM('Actuals'!F45:F46)");
formula(d,'D34',"='Assumptions'!F58");formula(d,'D35',"=SUM('Actuals'!F47:F48)-'Actuals'!F49");
formula(d,'D36',"=SUM('Actuals'!F50:F53)");formula(d,'D38','=D32+D33-SUM(D34:D35)+D36');
formula(d,'D39',"='Actuals'!F54");formula(d,'D40','=D38/D39');formula(d,'D41',"='Actuals'!F55");
formula(d,'D42','=D40/D41-1');formula(d,'D43','=D30/D32');d.getRange('D42:D43').setNumberFormat('0.0%');d.getRange('D40:D41').setNumberFormat('0.00');
band(d,46,'WACC / terminal growth sensitivity | EUR per share');
formula(d,'F48','=D40');d.getRange('G48:K48').values=[[.02,.025,.03,.035,.04]];
d.getRange('F49:F53').values=[[.075],[.085],[.095],[.105],[.115]];
d.dataTables.add('F48:K53',{rowInput:'D7',columnInput:'D6'});
d.getRange('G48:K48').setNumberFormat('0.0%');d.getRange('F49:F53').setNumberFormat('0.0%');d.getRange('G49:K53').setNumberFormat('0.00');
val(d,'C56','Debt estimate includes finance-lease repayments; not an exact Q3 debt disclosure.');
val(d,'C58','Q3 basic weighted shares proxy ending shares. Cash reserve is an analyst assumption.');
val(d,'C60','Stub cash flow is prorated; no separate October actuals are available.');
val(d,'C62','Non-operating assets use book values, without separate fair-value appraisal.');
d.getRange('C1:C80').format.columnWidth=60;

const peers=sheets.Peers;
title(peers,'Semiconductor equipment | Historical multiple comparison','4 Nov 2026 prices / FY2025 earnings and balance sheets | mixed fiscal year-ends');
peers.getRange('C5:J5').values=[['Company','Currency','Fiscal end','Revenue (mn)','Net income (mn)','Shares (mn)','Price','Cash incl ST (mn)']];
peers.getRange('L5:O5').values=[['Debt (mn)','Market cap (mn)','P/E','EV / sales']];
data.peers.forEach((p,i)=>{
 const r=6+i;peers.getRange(`C${r}:J${r}`).values=[[p.symbol,p.currency,p.fiscal_end,p.revenue,p.income,p.shares,p.price,p.cash]];val(peers,'L'+r,p.debt);
 formula(peers,'M'+r,`=H${r}*I${r}`);formula(peers,'N'+r,`=M${r}/G${r}`);formula(peers,'O'+r,`=(M${r}+L${r}-J${r})/F${r}`);
 val(peers,'C'+(15+i*3),`${p.symbol}: source FY2025 10-K/20-F; share data filed ${p.share_filed}`);
 val(peers,'D'+(16+i*3),`https://www.sec.gov/Archives/edgar/data/${p.symbol==='ASML.AS'?'937966':p.symbol==='AMAT'?'6951':p.symbol==='LRCX'?'707549':'319201'}/${p.accession.replaceAll('-','')}/`);
});
val(peers,'C30','These are historical-denominator diagnostics, not current TTM or forward trading comparables.');
val(peers,'C32','FY shares are split-adjusted where restated; subsequent buybacks are not captured.');
val(peers,'C34','Multiples need no FX conversion; absolute EUR and USD amounts must not be aggregated.');
val(peers,'C36','Peer cash excludes ST investments where not in the selected cash tag; EV/sales is approximate.');
val(peers,'C38','ASML is primarily lithography; AMAT, Lam and KLA have different equipment and service mixes.');
peers.getRange('L6:M9').setNumberFormat(fmt);peers.getRange('N6:O9').setNumberFormat('0.0"x"');
peers.getRange('C5:O5').format.rowHeight=34;peers.getRange('C5:O5').format.wrapText=true;

const su=sheets.Summary;
title(su,'ASML | Financial analysis and valuation','Independent learning project | prepared 5 November 2026 | EUR unless stated');
val(su,'C5','Selected case');formula(su,'F5',"='Assumptions'!D5");
for(const [r,l,f] of [[7,'DCF value per share',"='DCF'!D40"],[8,'Observed close (4 Nov 2026)',"='DCF'!D41"],[9,'Difference from observed close',"='DCF'!D42"],[11,'Enterprise value (EUR mn)',"='DCF'!D32"],[12,'Terminal share of enterprise value',"='DCF'!D43"],[14,'FY2025 revenue (EUR mn)',`='Actuals'!H${rows.Revenue}`],[15,'FY2025 reported FCF (EUR mn)',`='Actuals'!H${rows['Reported FCF']}`]]){val(su,'C'+r,l);formula(su,'F'+r,f);su.getRange('F'+r).format.font.color=ink;}
su.getRange('F7:F8').setNumberFormat('0.00');for(const r of [9,12])su.getRange('F'+r).setNumberFormat('0.0%');
val(su,'C18','Read the valuation as an assumption-dependent range, not a buy/sell recommendation.');
val(su,'C20','Controls: Assumptions D4 changes the case; DCF D6:D7 change WACC and terminal growth.');
val(su,'C22','Main estimates: normalised working capital, stub cash flow, debt bridge and share proxy.');
su.getRange('C26:F26').values=[['Year','Revenue','EBIT','Reported FCF']];
[2023,2024,2025].forEach((y,i)=>{const r=27+i,c=['F','G','H'][i];val(su,'C'+r,y);['Revenue','EBIT','Reported FCF'].forEach((k,j)=>formula(su,['D','E','F'][j]+r,`='Actuals'!${c}${rows[k]}`));});
const chart=su.charts.add('bar',su.getRange('C26:F29'));chart.title='Reported financial performance | EUR million';chart.setPosition('H5','P21');chart.titleTextStyle.typeface='Arial';chart.titleTextStyle.fontSize=13;
chart.xAxis={axisType:'textAxis',textStyle:{typeface:'Arial',fontSize:11}};chart.yAxis={numberFormatCode:'#,##0',numberFormatSourceLinked:false,textStyle:{typeface:'Arial',fontSize:11}};
for(const [s,r] of [[ac,'F5:H5'],[ass,'F5:J5'],[fc,'F5:J5'],[d,'F18:J18']])s.getRange(r).setNumberFormat('0');
su.getRange('D27:F29').format.font.color=ink;

// Independently calculated results catch shifted references and case-selector defects.
wb.recalculate();
const checks=[];
for(let i=0;i<3;i++){
 val(ass,'D4',i+1);wb.recalculate();
 const actual=d.getRange('D40').values[0][0], expected=data.results[['Base','Downside','Upside'][i]].per_share;
 if(Math.abs(actual-expected)>.01)throw new Error(`Case ${i}: Excel ${actual}, Python ${expected}`);
 checks.push({case:i+1,value:actual,expected});
}
val(ass,'D4',1);wb.recalculate();
for(let r=0;r<5;r++)for(let c=0;c<5;c++){
 const observed=d.getRangeByIndexes(48+r,6+c,1,1).values[0][0];
 if(Math.abs(observed-data.sensitivity[r][c])>.01)throw new Error(`Sensitivity mismatch ${r},${c}: ${observed}`);
}
const before=d.getRange('D40').values[0][0];const old=ass.getRange('J8').values[0][0];val(ass,'J8',old+.01);wb.recalculate();
if(d.getRange('D40').values[0][0]<=before)throw new Error('Late-year growth sensitivity did not recalculate');
val(ass,'J8',old);wb.recalculate();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!',options:{useRegex:true,maxResults:30},summary:'Formula error scan'});
await fs.writeFile(path.join(root,'reports/workbook_checks.json'),JSON.stringify({checks,errorScan:errors.ndjson},null,2));
console.log(JSON.stringify(checks));console.log(errors.ndjson);
await fs.mkdir(path.join(root,'reports/previews'),{recursive:true});
for(const name of names){const blob=await wb.render({sheetName:name,range:name==='Summary'?'C2:P30':name==='Assumptions'?'C2:J45':name==='Actuals'?'C2:J35':name==='Peers'?'C2:O12':'C2:K43',scale:1.4,format:'png'});await fs.writeFile(path.join(root,`reports/previews/${name}.png`),new Uint8Array(await blob.arrayBuffer()));}
const output=await SpreadsheetFile.exportXlsx(wb);await output.save(path.join(root,'model/ASML_Valuation.xlsx'));
