"""Portable standard-library verification of saved transfer metrics and regions."""
import json,csv,hashlib,math,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent
KEYS=['image_auroc','pixel_auroc','pixel_ap','aupro_005','normal_pixel_fpr']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def close(a,b,label):
    if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12):raise ValueError(f'{label}: {a} != {b}')
def stats(values):return {'mean':statistics.mean(values),'sample_sd':statistics.stdev(values)}
def main():
    manifest=read(HERE/'manifest.json')
    for name,expected in manifest['files'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected,'File hash mismatch: '+name
    data=read(HERE/'metrics.json');seeds=manifest['seeds'];conditions=manifest['conditions']
    assert len(data)==3 and sorted(r['seed'] for r in data)==seeds
    data=sorted(data,key=lambda r:r['seed']);numeric_widths={'pixel_ap':[],'aupro_005':[]}
    for row in data:
        for endpoint in ['initial','final']:
            record=row[endpoint];assert record['native_grid']==manifest['native_grid'] and record['development_only']
            assert sorted(record['conditions'])==sorted(conditions)
            for condition in conditions:
                values=record['conditions'][condition]
                for key in KEYS:assert math.isfinite(values[key]) and 0<=values[key]<=1,(condition,key)
                for point,lo,hi in [('pixel_ap','pixel_ap_lower','pixel_ap_upper'),('aupro_005','aupro_lower','aupro_upper')]:
                    assert 0<=values[lo]<=values[point]+1e-12 and values[point]<=values[hi]+1e-12<=1+1e-12
                    numeric_widths[point].append(values[hi]-values[lo])
            for key in KEYS:close(record['macro'][key],statistics.mean(record['conditions'][c][key] for c in conditions),'macro')
    recomputed={}
    for scope in ['macro',*conditions]:
        recomputed[scope]={}
        for key in KEYS:
            series={e:[r[e]['macro'][key] if scope=='macro' else r[e]['conditions'][scope][key] for r in data] for e in ['initial','final']}
            result={'before':stats(series['initial']),'after':stats(series['final']),
                    'paired_delta':stats([b-a for a,b in zip(series['initial'],series['final'])])}
            for endpoint in result:
                for field,value in result[endpoint].items():close(value,manifest['expected_metrics'][scope][key][endpoint][field],f'{scope}/{key}/{endpoint}/{field}')
            recomputed[scope][key]=result
    rows=list(csv.DictReader((HERE/'paired-regions.csv').open(newline='',encoding='utf-8')));populations={};region_counts={}
    assert set(int(r['seed']) for r in rows)==set(seeds)
    for seed in seeds:
        selected=[r for r in rows if int(r['seed'])==seed];ids={r['region_id'] for r in selected};assert len(ids)==len(selected)
        population={r['region_id']:(r['condition'],r['defect'],int(r['area']),int(r['small'])) for r in selected}
        populations[seed]=population
        assert all(r['condition'] in conditions for r in selected)
        for r in selected:
            assert int(r['small'])==int(int(r['area'])<=manifest['small_area_fraction']*math.prod(manifest['native_grid']))
            for e in ['initial','final']:
                coverage=float(r[e+'_coverage']);assert 0<=coverage<=1
                assert int(r[e+'_detected'])==int(coverage>=manifest['minimum_region_coverage'])
        region_counts[seed]={}
        for group in ['all','small','fracture']:
            subset=[r for r in selected if group=='all' or (group=='small' and int(r['small'])) or (group=='fracture' and r['defect']=='fracture')]
            before=sum(int(r['initial_detected']) for r in subset);after=sum(int(r['final_detected']) for r in subset)
            lost=sum(int(r['initial_detected']) and not int(r['final_detected']) for r in subset)
            gained=sum(not int(r['initial_detected']) and int(r['final_detected']) for r in subset)
            assert after-before==gained-lost
            result={'total':len(subset),'before_detected':before,'after_detected':after,'lost':lost,'gained':gained,
                    'before_recall':before/len(subset) if subset else None,'after_recall':after/len(subset) if subset else None}
            for key,value in result.items():
                expected=manifest['expected_regions'][str(seed)][group][key]
                if value is None:assert expected is None
                else:close(value,expected,f'{seed}/{group}/{key}')
            region_counts[seed][group]=result
    assert all(populations[s]==populations[seeds[0]] for s in seeds)
    result={'status':'PASS_SAVED_TRANSFER_AGGREGATION','seeds':seeds,'conditions':conditions,'region_rows':len(rows),
            'region_counts':region_counts,'max_numerical_interval_width_pp':{k:max(v)*100 for k,v in numeric_widths.items()},
            'scope':manifest['scope']}
    print(json.dumps(result))
if __name__=='__main__':main()
