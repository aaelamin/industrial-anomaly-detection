"""Readable AeBAD tables from the verified, saved development evidence only."""
import argparse, csv, hashlib, json, statistics, subprocess, sys
from pathlib import Path

KEYS = ['image_auroc', 'pixel_auroc', 'pixel_ap', 'aupro_005', 'normal_pixel_fpr']
LABELS = ['Image AUROC', 'Pixel AUROC', 'Pixel AP', 'PRO (5% FPR)', 'Normal-pixel FPR']

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); bundle = args.bundle.resolve(); out = args.output.resolve()
    assert not out.exists(), 'Choose a new output folder; retain prior exports'
    checked = subprocess.run([sys.executable, '-B', str(bundle/'verify.py')],
                             cwd=bundle.parent, capture_output=True, text=True, check=True)
    receipt = json.loads(checked.stdout)
    assert receipt['status'] == 'PASS_SAVED_TRANSFER_AGGREGATION'
    manifest = json.loads((bundle/'manifest.json').read_text())
    data = sorted(json.loads((bundle/'metrics.json').read_text()), key=lambda r:r['seed'])
    assert [r['seed'] for r in data] == [0, 1, 2]
    out.mkdir(parents=True)
    def save(name, rows):
        with (out/name).open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    rows = []
    for scope in ['macro', *manifest['conditions']]:
        for key in KEYS:
            value = manifest['expected_metrics'][scope][key]
            row = {'scope':scope, 'metric':key}
            for phase in ['before', 'after', 'paired_delta']:
                for statistic in ['mean', 'sample_sd']:
                    row[phase+'_'+statistic+'_percent_or_pp'] = 100*value[phase][statistic]
            rows.append(row)
    save('endpoint-results.csv', rows)
    raw = []
    for r in data:
        for phase in ['initial', 'final']:
            for scope in ['macro', *manifest['conditions']]:
                v = r[phase]['macro'] if scope == 'macro' else r[phase]['conditions'][scope]
                raw.append({'seed':r['seed'], 'endpoint':phase, 'scope':scope,
                            **{k+'_percent':100*v[k] for k in KEYS}})
    save('seed-endpoints.csv', raw)
    regions = []
    for seed in range(3):
        for group in ['all', 'small', 'fracture']:
            regions.append({'seed':seed, 'subset':group, **receipt['region_counts'][str(seed)][group]})
    save('region-transitions.csv', regions)
    save('normal-calibration-thresholds.csv', [
        {'seed':r['seed'], 'endpoint':e, 'pixel_threshold':r[e]['pixel_threshold'],
         'threshold_bin_width':r[e]['threshold_bin_width']} for r in data for e in ['initial', 'final']])
    lines = ['# AeBAD development tables for the report team', '',
      'Three seeds on the same declared 160 development images. Initialization is a diagnostic control, not a competitive method. Mean and sample SD describe seed variation. Macro scores weight the four capture-condition groups equally within each seed.', '',
      '| Metric | Initialization (%) | Epoch 500 (%) | Paired change (points) |',
      '|---|---:|---:|---:|']
    for key, label in zip(KEYS, LABELS):
        v=manifest['expected_metrics']['macro'][key]
        values=[f"{100*v[e]['mean']:.2f} ± {100*v[e]['sample_sd']:.2f}" for e in ['before','after','paired_delta']]
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines += ['', 'Higher is better for ranking metrics; lower is better for false-positive rate. Pixel AP and PRO use the frozen native-grid histogram evaluation. Numerical approximation bounds are in the verification receipt and are not statistical confidence intervals.', '',
      '## Epoch-500 capture-condition breakdown', '',
      '| Provider group | '+' | '.join(label+' (%)' for label in LABELS)+' |',
      '|---|'+'---:|'*len(KEYS)]
    for scope in manifest['conditions']:
        values=[manifest['expected_metrics'][scope][k]['after'] for k in KEYS]
        lines.append('| '+scope+' | '+' | '.join(f"{100*v['mean']:.2f} ± {100*v['sample_sd']:.2f}" for v in values)+' |')
    lines += ['', 'These are provider folder names. In particular, `same` is not a certificate of matched train/test capture conditions or physical-object independence.', '',
      '## Region transitions at separately calibrated normal-only thresholds', '',
      '| Subset | Regions per seed | Initially detected (seeds 0 / 1 / 2) | Finally detected | Lost | Gained |',
      '|---|---:|---|---|---|---|']
    for group in ['all','small','fracture']:
        selected=[r for r in regions if r['subset']==group]
        assert len({r['total'] for r in selected})==1
        values=[' / '.join(str(r[k]) for r in selected) for k in ['before_detected','after_detected','lost','gained']]
        lines.append(f"| {group} | {selected[0]['total']} | "+' | '.join(values)+' |')
    lines += ['',
      'Small means at most 0.1% of native image area. A region is detected when at least 10% of its pixels exceed the normal-only calibrated threshold. Small and fracture subsets overlap. Gains do not erase losses; net gains must be interpreted alongside the increased normal-pixel false-positive rate. Nondefect mask pixels include intact objects as well as physical background.', '',
      'The method is refitted on 128 new-domain training normals and calibrated on 32 separate training normals. This is a domain application, not zero-shot transfer of learned MVTec flow weights. Dataset-path separation is verified; physical-blade and acquisition-session independence are not. The 1,479 reserved AeBAD images remain excluded.', '',
      'The CSVs retain unrounded values, all seed/condition endpoints, region transitions and the recorded calibration thresholds. The exporter loads only saved JSON/CSV evidence and runs its standard-library verifier. It performs no inference, metric rescoring or threshold selection.']
    (out/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    result={'status':'PASS_VERIFIED_TRANSFER_TABLE_EXPORT', 'endpoint_rows':len(rows), 'seed_rows':len(raw),
            'region_rows':len(regions), 'source_manifest_sha256':sha(bundle/'manifest.json'),
            'source_verifier':receipt, 'new_predictions':0, 'source_images_opened':0,
            'files':{f.name:sha(f) for f in out.iterdir() if f.is_file()}}
    (out/'table-checks.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','endpoint_rows','seed_rows','region_rows']}))

if __name__ == '__main__': main()
