"""Download the public source set; keep retrieval dates and file hashes."""
from pathlib import Path
from datetime import datetime, timezone
import urllib.request
import json
import hashlib

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data/raw'
SOURCES={
 'annual_2025.html':'https://www.sec.gov/Archives/edgar/data/937966/000162828026011378/asml-20251231.htm',
 'annual_2024.html':'https://www.sec.gov/Archives/edgar/data/937966/000093796625000009/asml-20241231.htm',
 'q3_2026.html':'https://www.sec.gov/Archives/edgar/data/937966/000162828026065813/financial_statementsxusx.htm',
 'q4_2025.pdf':'https://ourbrand.asml.com/m/fded19d66e819e4b/original/Financial-statements-US-GAAP-Q4-2025.pdf',
 'amat_facts.json':'https://data.sec.gov/api/xbrl/companyfacts/CIK0000006951.json',
 'lam_facts.json':'https://data.sec.gov/api/xbrl/companyfacts/CIK0000707549.json',
 'kla_facts.json':'https://data.sec.gov/api/xbrl/companyfacts/CIK0000319201.json',
}
for ticker in ['ASML.AS','AMAT','LRCX','KLAC']:
 SOURCES[ticker+'_price.json']=f'https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1=1793664000&period2=1793836800&interval=1d'

def main():
 RAW.mkdir(parents=True,exist_ok=True)
 manifest=[]
 for name,url in SOURCES.items():
  target=RAW/name
  if not target.exists():
   request=urllib.request.Request(url,headers={'User-Agent':'Independent financial research contact shubham.nanewar@gmail.com'})
   with urllib.request.urlopen(request,timeout=60) as response:
    target.write_bytes(response.read())
  content=target.read_bytes()
  manifest.append(dict(file=name,url=url,sha256=hashlib.sha256(content).hexdigest(),bytes=len(content),
                       checked_at=datetime.now(timezone.utc).isoformat(),cached=True))
 (RAW/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(f'Checked {len(manifest)} cached sources')

if __name__=='__main__':
 main()
