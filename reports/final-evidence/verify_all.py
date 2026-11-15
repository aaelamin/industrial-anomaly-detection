"""Verify the handoff files, local links and both saved-number bundles."""
import csv,hashlib,json,math,re,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
    manifest=json.loads((HERE/'delivery-manifest.json').read_text(encoding='utf-8'))
    assert manifest['mvtec_runs']==315 and manifest['aebad_runs']==3 and manifest['reserved_images_opened']==0
    files=manifest['files'];observed={p.relative_to(HERE).as_posix() for p in HERE.rglob('*') if p.is_file() and p.name!='delivery-manifest.json' and '__pycache__' not in p.parts}
    assert observed==set(files),'Delivered file set differs'
    for name,h in files.items():
        p=(HERE/name).resolve();assert p.is_relative_to(HERE.resolve()) and hashlib.sha256(p.read_bytes()).hexdigest()==h,'File hash mismatch: '+name
    links=0
    # Check handoff navigation and gallery image references, not literal CLI/code examples.
    for p in [HERE/'README.md',*(HERE/'galleries').rglob('index.html'),*(HERE/'galleries').rglob('HEATMAPS.md'),HERE/'matched-gallery/index.html',HERE/'matched-gallery/HEATMAPS.md']:
        t=p.read_text(encoding='utf-8');targets=re.findall(r'\]\(([^)]+)\)',t) if p.suffix=='.md' else re.findall(r'(?:src|href)="([^"]+)"',t)
        for target in targets:
            if target.startswith(('https:','http:','#')):continue
            path=(p.parent/target.split('#')[0]).resolve();assert path.is_relative_to(HERE.resolve()) and path.exists(),(p.name,target);links+=1
    checks=[]
    for folder in ['evidence-bundle','transfer-evidence']:
        result=subprocess.run([sys.executable,'-B',str(HERE/folder/'verify.py')],cwd=HERE.parent,capture_output=True,text=True,check=True)
        checks.append(json.loads(result.stdout)['status'])
    assert checks==['PASS_SAVED_METRIC_AGGREGATION','PASS_SAVED_TRANSFER_AGGREGATION']
    # Independently check that the writers' exported cells match the verified bundles.
    def rows(path):
        with path.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
    cells=0
    def close(a,b):
        nonlocal cells
        assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12),(a,b)
        cells+=1
    def key(r):return r['model'],r['category'],int(r['seed'])
    source=rows(HERE/'evidence-bundle/original-per_run.csv')+rows(HERE/'evidence-bundle/additional-per-run.csv')
    source={key(r):r for r in source};export=rows(HERE/'team-tables/all-315-runs.csv')
    assert len(source)==len(export)==315 and set(source)=={key(r) for r in export}
    for r in export:
        for metric in ['image_auroc','pixel_auroc','pixel_average_precision']:close(r[metric],source[key(r)][metric])
    bench=json.loads((HERE/'evidence-bundle/manifest.json').read_text())
    macro=rows(HERE/'team-tables/main-results.csv');assert len(macro)==7
    for r in macro:
        for metric in ['image_auroc','pixel_auroc','pixel_average_precision']:
            reference=bench['expected_overall'][r['model']][metric]
            close(r[metric+'_mean_percent'],100*reference['mean'])
            close(r[metric+'_sample_sd_percent'],100*reference['std'])
    transfer=json.loads((HERE/'transfer-evidence/manifest.json').read_text())
    endpoints=rows(HERE/'transfer-team-tables/endpoint-results.csv');assert len(endpoints)==25
    for r in endpoints:
        reference=transfer['expected_metrics'][r['scope']][r['metric']]
        for phase in ['before','after','paired_delta']:
            for statistic in ['mean','sample_sd']:close(r[phase+'_'+statistic+'_percent_or_pp'],100*reference[phase][statistic])
    transitions=rows(HERE/'transfer-team-tables/region-transitions.csv');assert len(transitions)==9
    for r in transitions:
        reference=transfer['expected_regions'][r['seed']][r['subset']]
        for k,v in reference.items():close(r[k],v)
    seeds={r['seed']:r for r in json.loads((HERE/'transfer-evidence/metrics.json').read_text())}
    exported_seeds=rows(HERE/'transfer-team-tables/seed-endpoints.csv');assert len(exported_seeds)==30
    for r in exported_seeds:
        endpoint=seeds[int(r['seed'])][r['endpoint']]
        reference=endpoint['macro'] if r['scope']=='macro' else endpoint['conditions'][r['scope']]
        for metric in ['image_auroc','pixel_auroc','pixel_ap','aupro_005','normal_pixel_fpr']:close(r[metric+'_percent'],100*reference[metric])
    print(json.dumps({'status':'PASS_DELIVERED_FILES_LINKS_AND_NUMBERS','files':len(files),'local_links':links,'mvtec_runs':315,'aebad_seeds':3,'numeric_checks':checks,'exported_numeric_cells_checked':cells,'model_execution':False,'source_images_opened':0}))
if __name__=='__main__':main()
