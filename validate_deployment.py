"""Validate a deployed merchant-monitor outputs/ folder without changing evaluated files.
Run: python validate_deployment.py
Uses only Python standard library.
"""
import csv, gzip, hashlib, json
from decimal import Decimal
from pathlib import Path

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'
EXPECTED_FILES=[
 'daily_sales_prepared.csv','business_daily_prepared.csv','merchant_risk.csv',
 'predictions_original_isolation_forest.csv','daily_sales_new_prepared.csv',
 'business_daily_new_prepared.csv','predictions_new_isolation_forest.csv',
 'predictions_new_random_forest.csv','predictions_new_gradient_boosting.csv',
 'predictions_new_logistic_regression.csv','transactions_original_captured.csv.gz',
 'transactions_new_captured.csv.gz'
]
EXPECTED_COUNTS={'predictions_original_isolation_forest.csv':11,
 'predictions_new_isolation_forest.csv':13,'predictions_new_random_forest.csv':13,
 'predictions_new_gradient_boosting.csv':13,'predictions_new_logistic_regression.csv':15}

def csv_rows(path):
    opener=gzip.open if path.suffix=='.gz' else open
    with opener(path,'rt',newline='',encoding='utf-8-sig') as handle:
        yield from csv.DictReader(handle)

def cents(x):return int((Decimal(str(x))*100).quantize(Decimal('1')))

def audit(portfolio):
    suffix='_new' if portfolio=='new' else ''
    daily=OUT/('daily_sales'+suffix+'_prepared.csv')
    category=OUT/('business_daily'+suffix+'_prepared.csv')
    tx=OUT/('transactions_'+portfolio+'_captured.csv.gz')
    want_count=481350 if portfolio=='new' else 482835
    want_sales=15797921737 if portfolio=='new' else 15837198740
    ds=[];dates=set();merchants=set();day_sales=0;day_tx=0
    for row in csv_rows(daily):
        ds.append(row)
        dates.add(row['Date']);merchants.add(row['Merchant'])
        day_sales+=cents(row['Daily_Sales']);day_tx+=int(row['Transaction_Count'])
    assert len(ds)==9450 and len(dates)==90 and len(merchants)==105
    assert day_sales==want_sales and day_tx==want_count
    cat_rows=0;category_total=0
    for row in csv_rows(category):
        cat_rows+=1;category_total+=cents(row['Business_Sales'])
    assert cat_rows==47250 and category_total==want_sales
    tx_rows=0;tx_total=0;txmerchants=set()
    for row in csv_rows(tx):
        assert row.get('Status')=='Captured'
        tx_rows+=1;tx_total+=cents(row['Amount']);txmerchants.add(row['Merchant'])
    assert tx_rows==want_count and tx_total==want_sales and txmerchants==merchants
    print(f'PASS {portfolio.upper()}: 105 merchants | 90 days | 9450 merchant-days | 47250 category-days | {tx_rows:,} Captured transactions | USD {tx_total/100:,.2f}')
    return merchants

def main():
    missing=[name for name in EXPECTED_FILES if not (OUT/name).is_file()]
    if missing:raise SystemExit('Missing outputs: '+', '.join(missing))
    names={'original':audit('original'),'new':audit('new')}
    for name, flagged in EXPECTED_COUNTS.items():
        data=list(csv_rows(OUT/name))
        universe='original' if 'original' in name else 'new'
        assert len(data)==105
        assert {r['Merchant'] for r in data}==names[universe]
        assert all(r['Prediction'] in {'0','1'} for r in data)
        got=sum(int(r['Prediction']) for r in data)
        assert got==flagged,(name,got,flagged)
        print('PASS',name,'105 merchant predictions; flagged',got)
    manifest=HERE/'DATA_MANIFEST.json'
    if manifest.is_file():
        doc=json.loads(manifest.read_text())
        for name,meta in doc['files'].items():
            p=HERE/name
            assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256'],name
        print('PASS data manifest hashes (user uploads retained without modification)')
    print('PASS ALL CHECKS')

if __name__=='__main__':main()
