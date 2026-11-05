"""Rebuild the source extracts and an independent DCF calculation."""
from pathlib import Path
from datetime import date, datetime, timezone
import json
import csv
from lxml import html

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/raw'
OUT = ROOT / 'data/processed'


def extract_facts(filename):
    tree = html.fromstring((RAW / filename).read_bytes())
    contexts = {}
    for node in tree.xpath('//*[contains(local-name(),"context")]'):
        if not node.get('id'):
            continue
        text = ''.join(node.itertext())
        periods = node.xpath('.//*[contains(local-name(),"period")]')
        dates = list(periods[0].itertext()) if periods else []
        contexts[node.get('id')] = (dates, not node.xpath('.//*[contains(local-name(),"segment") or contains(local-name(),"scenario")]'))
    facts = {}
    precision = {}
    for node in tree.xpath('//*[contains(local-name(),"nonfraction")]'):
        ctx = contexts.get(node.get('contextref'))
        if not ctx or not ctx[1]:
            continue
        text = ''.join(node.itertext()).replace(',', '').strip()
        if text in ['—', '–', '-']:
            value = 0.0
        else:
            try:
                value = float(text)
            except ValueError:
                continue
        value *= 10 ** int(node.get('scale', 0))
        value *= -1 if node.get('sign') == '-' else 1
        key = (node.get('name'), tuple(ctx[0]))
        decimals = int(node.get('decimals', '-9')) if node.get('decimals') != 'INF' else 9
        if key in facts and decimals < precision[key]:
            continue
        if key in facts and decimals == precision[key] and abs(facts[key] - value) > max(1, 10**(-decimals)):
            raise ValueError(f'Conflicting consolidated fact: {key}')
        facts[key] = value
        precision[key] = decimals
    return facts


TAGS = {
    'Revenue': 'us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax',
    'Cost of sales': 'us-gaap:CostOfGoodsAndServicesSold',
    'Gross profit': 'us-gaap:GrossProfit',
    'R&D': 'us-gaap:ResearchAndDevelopmentExpense',
    'SG&A': 'us-gaap:SellingGeneralAndAdministrativeExpense',
    'EBIT': 'us-gaap:OperatingIncomeLoss',
    'Net income': 'us-gaap:NetIncomeLoss',
    'Operating cash flow': 'us-gaap:NetCashProvidedByUsedInOperatingActivities',
    'Cash capex PPE': 'us-gaap:PaymentsToAcquirePropertyPlantAndEquipment',
    'Cash capex intangibles': 'us-gaap:PaymentsToAcquireIntangibleAssets',
    'D&A': 'us-gaap:DepreciationDepletionAndAmortization',
    'Cash': 'us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents',
    'Short-term investments': 'us-gaap:AvailableForSaleSecuritiesDebtSecuritiesCurrent',
    'Current debt': 'us-gaap:LongTermDebtCurrent',
    'Long-term debt': 'us-gaap:LongTermDebtNoncurrent',
    'Receivables': 'us-gaap:AccountsReceivableNetCurrent',
    'Inventories': 'us-gaap:InventoryNet',
    'Contract assets': 'us-gaap:ContractWithCustomerAssetNetCurrent',
    'Payables': 'us-gaap:AccountsPayableCurrent',
    'Current customer advances': 'us-gaap:ContractWithCustomerLiabilityCurrent',
    'Noncurrent customer advances': 'us-gaap:ContractWithCustomerLiabilityNoncurrent',
    'Assets': 'us-gaap:Assets',
    'Liabilities': 'us-gaap:Liabilities',
    'Equity': 'us-gaap:StockholdersEquity',
    'Shares outstanding': 'us-gaap:CommonStockSharesOutstanding',
}


def historical():
    latest = extract_facts('annual_2025.html')
    previous = extract_facts('annual_2024.html')
    result = {}
    for year in [2023, 2024, 2025]:
        record = {}
        for label, tag in TAGS.items():
            duration = (f'{year}-01-01', f'{year}-12-31')
            instant = (f'{year}-12-31',)
            found = None
            for data in [latest, previous]:
                for period in [duration, instant]:
                    if (tag, period) in data:
                        found = data[tag, period] / 1e6
                        break
                if found is not None:
                    break
            if found is None:
                raise KeyError(f'Missing source fact: {label}, {year}, {tag}')
            record[label] = found
        record['Reported FCF'] = record['Operating cash flow'] - record['Cash capex PPE'] - record['Cash capex intangibles']
        record['Core NWC incl advances'] = sum(record[k] for k in ['Receivables', 'Inventories', 'Contract assets']) - sum(record[k] for k in ['Payables', 'Current customer advances', 'Noncurrent customer advances'])
        result[str(year)] = record
    return result


def quoted_price(symbol):
    data = json.loads((RAW / f'{symbol}_price.json').read_text())['chart']['result'][0]
    for stamp, close in zip(data['timestamp'], data['indicators']['quote'][0]['close']):
        if datetime.fromtimestamp(stamp, timezone.utc).date().isoformat() == '2026-11-04':
            return close, data['meta']['currency']
    raise ValueError(f'Missing dated close for {symbol}')


def peer_data(hist):
    result = []
    for symbol, filename in [('AMAT', 'amat_facts.json'), ('LRCX', 'lam_facts.json'), ('KLAC', 'kla_facts.json')]:
        data = json.loads((RAW / filename).read_text())
        source = data['facts']['us-gaap']
        def pick(tags):
            for tag in tags:
                candidates = [v for entries in source.get(tag, {}).get('units', {}).values() for v in entries
                              if v.get('form') == '10-K' and v.get('end', '').startswith('2025')
                              and v.get('filed', '') <= '2026-11-04'
                              and ('start' not in v or (date.fromisoformat(v['end']) - date.fromisoformat(v['start'])).days > 300)]
                if candidates:
                    value = sorted(candidates, key=lambda v: (v['end'], v['filed']))[-1]
                    return value['val'] / 1e6, value
            raise KeyError(f'{symbol}: {tags}')
        revenue, record = pick(['RevenueFromContractWithCustomerExcludingAssessedTax', 'Revenues'])
        income, _ = pick(['NetIncomeLoss'])
        shares, share_record = pick(['CommonStockSharesOutstanding'])
        cash, _ = pick(['CashAndCashEquivalentsAtCarryingValue'])
        debt, _ = pick(['LongTermDebt', 'LongTermDebtNoncurrent'])
        if 'LongTermDebt' not in source:
            current, _ = pick(['LongTermDebtCurrent'])
            debt += current
        price, currency = quoted_price(symbol)
        result.append(dict(symbol=symbol, currency=currency, fiscal_end=record['end'], revenue=revenue,
                           income=income, shares=shares, cash=cash, debt=debt, price=price,
                           share_filed=share_record['filed'], accession=record['accn']))
    a = hist['2025']
    result.insert(0, dict(symbol='ASML.AS', currency='EUR', fiscal_end='2025-12-31', revenue=a['Revenue'],
                         income=a['Net income'], shares=a['Shares outstanding'], cash=a['Cash'] + a['Short-term investments'],
                         debt=a['Current debt'] + a['Long-term debt'], price=quoted_price('ASML.AS')[0],
                         share_filed='2026-02-25', accession='0001628280-26-011378'))
    return result


ASSUMPTIONS = {
    'growth': [[.08, .08, .07, .06, .05], [.02, .03, .03, .03, .03], [.12, .12, .10, .08, .06]],
    'gross_margin': [[.56, .565, .57, .575, .58], [.53, .53, .535, .54, .54], [.575, .59, .60, .61, .62]],
    'rd': [[.105]*5, [.115]*5, [.10]*5],
    'sga': [[.032]*5, [.035]*5, [.03]*5],
    'tax': [[.18]*5, [.19]*5, [.18]*5],
    'da': [[.025]*5]*3,
    'capex': [[.045]*5, [.05]*5, [.045]*5],
    'nwc': [[.10]*5]*3,
}


def dcf(case=0, wacc=.095, terminal_growth=.03, inputs=None):
    if wacc <= terminal_growth:
        raise ValueError('Discount rate must exceed terminal growth')
    assumptions = inputs or ASSUMPTIONS
    revenue, nwc = 44500., 4450.
    forecast = []
    days = (date(2026, 12, 31) - date(2026, 11, 4)).days
    stub = days / 365
    # The Q4 forecast is prorated because October cash flows have not been separately disclosed.
    q4_revenue = 44500 - 29755.4
    q4_ebit = q4_revenue * .57 - 1200 - 400
    q4_fcff = q4_ebit * (1 - .17) + 330 - 650 - 500
    remaining_fcff = q4_fcff * days / (date(2026, 12, 31) - date(2026, 9, 27)).days
    pv_stub = remaining_fcff / (1 + wacc)**stub
    for i in range(5):
        revenue *= 1 + assumptions['growth'][case][i]
        gp = revenue * assumptions['gross_margin'][case][i]
        rd = revenue * assumptions['rd'][case][i]
        sga = revenue * assumptions['sga'][case][i]
        ebit = gp - rd - sga
        nopat = ebit * (1 - assumptions['tax'][case][i])
        da = revenue * assumptions['da'][case][i]
        capex = revenue * assumptions['capex'][case][i]
        new_nwc = revenue * assumptions['nwc'][case][i]
        fcff = nopat + da - capex - (new_nwc - nwc)
        exponent = i + 1 + stub
        forecast.append(dict(year=2027+i, revenue=revenue, gp=gp, rd=rd, sga=sga, ebit=ebit,
                             nopat=nopat, da=da, capex=capex, nwc=new_nwc, delta_nwc=new_nwc-nwc,
                             fcff=fcff, exponent=exponent, pv=fcff/(1+wacc)**exponent))
        nwc = new_nwc
    # Terminal cash flow is rebuilt from operating drivers, not the last FCF multiplied by growth.
    last = forecast[-1]
    terminal_revenue = revenue * (1+terminal_growth)
    terminal_nopat = terminal_revenue * (assumptions['gross_margin'][case][-1] - assumptions['rd'][case][-1] - assumptions['sga'][case][-1]) * (1-assumptions['tax'][case][-1])
    terminal_fcff = terminal_nopat + terminal_revenue*(assumptions['da'][case][-1]-assumptions['capex'][case][-1]) - terminal_growth * nwc
    terminal = terminal_fcff / (wacc-terminal_growth)
    terminal_pv = terminal / (1+wacc)**last['exponent']
    ev = pv_stub + sum(v['pv'] for v in forecast) + terminal_pv
    # Latest disclosed debt is estimated from the annual balance less nine-month repayments.
    debt = 1681.9 + 2709.0 - 1693.7
    cash = 8030.9 + 4704.2
    non_operating_assets = 2239.4 + 1129.2 + 303.0 + 1782.5
    equity = ev + cash - 2000 - debt + non_operating_assets
    return dict(forecast=forecast, pv_stub=pv_stub, q4_fcff=q4_fcff, remaining_fcff=remaining_fcff,
                terminal_fcff=terminal_fcff, terminal_pv=terminal_pv, ev=ev, equity=equity,
                per_share=equity/383.1, terminal_share=terminal_pv/ev,
                wacc=wacc, terminal_growth=terminal_growth)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    hist = historical()
    peers = peer_data(hist)
    results = {name: dcf(i) for i, name in enumerate(['Base', 'Downside', 'Upside'])}
    dataset = dict(historical=hist, peers=peers, assumptions=ASSUMPTIONS, results=results,
                   valuation_date='2026-11-04', source_cutoff='2026-11-04',
                   market_price=quoted_price('ASML.AS')[0])
    dataset['sensitivity'] = [[dcf(0,w,g)['per_share'] for g in [.02,.025,.03,.035,.04]] for w in [.075,.085,.095,.105,.115]]
    (OUT/'model_data.json').write_text(json.dumps(dataset, indent=2))
    for name, rows in [('historical', [dict(year=k, **v) for k,v in hist.items()]), ('peers', peers), ('base_forecast', results['Base']['forecast'])]:
        with (OUT/f'{name}.csv').open('w', newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps({k:round(v['per_share'],2) for k,v in results.items()}))


if __name__ == '__main__':
    main()
