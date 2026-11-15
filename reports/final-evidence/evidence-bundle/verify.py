"""Recompute report aggregation using only the Python standard library.

Run from any directory: python /path/to/evidence-bundle/verify.py
This verifies saved metrics, not training or inference from raw images.
"""
from pathlib import Path
import argparse, csv, hashlib, itertools, json, math, statistics

HERE = Path(__file__).resolve().parent
METRICS = ('image_auroc','pixel_auroc','pixel_average_precision')

def close(a,b,label):
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12):
        raise ValueError(f'{label}: computed {a!r}, expected {b!r}')

def verify():
    manifest=json.loads((HERE/'manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        actual=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('File hash mismatch: '+name)
    data={}
    for name in ('original-per_run.csv','additional-per-run.csv'):
        with (HERE/name).open(newline='',encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                key=(row['model'],row['category'],int(row['seed']))
                if key in data:raise ValueError('Duplicate row: '+str(key))
                scores={k:float(row[k]) for k in METRICS}
                if not all(math.isfinite(v) and 0<=v<=1 for v in scores.values()):
                    raise ValueError('Invalid metric: '+str(key))
                data[key]=scores
    grid=set(itertools.product(manifest['models'],manifest['categories'],manifest['seeds']))
    if set(data)!=grid:raise ValueError('Complete model/category/seed grid mismatch')
    overall={};per_category={}
    for model in manifest['models']:
        overall[model]={};per_category[model]={}
        for k in METRICS:
            seeds=[statistics.mean(data[model,c,s][k] for c in manifest['categories']) for s in manifest['seeds']]
            result={'mean':statistics.mean(seeds),'std':statistics.stdev(seeds),'seed_means':seeds}
            expected=manifest['expected_overall'][model][k]
            for field in ('mean','std'):close(result[field],expected[field],f'{model}/{k}/{field}')
            for a,b in zip(seeds,expected['seed_means']):close(a,b,f'{model}/{k}/seed')
            overall[model][k]=result
        for c in manifest['categories']:
            per_category[model][c]={k:{'mean':statistics.mean(data[model,c,s][k] for s in manifest['seeds']),
                'std':statistics.stdev(data[model,c,s][k] for s in manifest['seeds'])} for k in METRICS}
    changes={}
    with (HERE/'original-paired_changes.csv').open(newline='',encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            c,k=row['category'],row['metric']
            vals=[100*(data['fusion',c,s][k]-data['middle',c,s][k]) for s in manifest['seeds']]
            for s,v in zip(manifest['seeds'],vals):close(v,float(row[f'seed_{s}']),f'paired/{c}/{k}/{s}')
            close(statistics.mean(vals),float(row['mean_change_points']),f'paired/{c}/{k}/mean')
            changes[c,k]=vals
    if set(changes)!=set(itertools.product(manifest['categories'],METRICS)):
        raise ValueError('Paired-change coverage mismatch')
    paired={}
    for k in METRICS:
        vals=[statistics.mean(changes[c,k][i] for c in manifest['categories']) for i in range(3)]
        paired[k]={'mean_change_pp':statistics.mean(vals),'sample_sd_pp':statistics.stdev(vals),'seed_changes_pp':vals}
    return {'status':'PASS_SAVED_METRIC_AGGREGATION','rows':len(data),'models':manifest['models'],
        'categories':manifest['categories'],'seeds':manifest['seeds'],'overall':overall,'per_category':per_category,
        'fusion_minus_middle':paired,'scope':manifest['scope'],'uncertainty':manifest['uncertainty']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path);args=p.parse_args()
    result=verify()
    if args.output_dir:
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'verified-results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        lines=['# Verified saved-metric aggregation','',result['scope'],'',
            '| Model | Image AUROC | Pixel AUROC | Pixel AP |','|---|---:|---:|---:|']
        for model,m in result['overall'].items():
            lines.append('| '+model+' | '+' | '.join(f"{100*m[k]['mean']:.2f} ± {100*m[k]['std']:.2f}" for k in METRICS)+' |')
        lines+=['','All entries: percentages, mean and sample SD over three seed-level category means.','',
            'Fusion minus middle-control pixel AP: '+str(result['fusion_minus_middle']['pixel_average_precision']),
            '', 'This does not reproduce training, inference, pixel-metric construction, or raw-image access audits.']
        (args.output_dir/'verified-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'rows':result['rows'],'models':result['models'],
        'fusion_minus_middle_ap_pp':result['fusion_minus_middle']['pixel_average_precision']['mean_change_pp']}))
if __name__=='__main__':main()
