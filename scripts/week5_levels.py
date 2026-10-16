"""Inspect the saved Week 4 models one feature level at a time."""
import csv
import itertools
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.week5_model import seed_all, torch, np
from scripts.week4_colab import restore_model, write_json, identity, digest
from week3 import build_model, MVTecDataset, DataLoader, file_hash
from sklearn.metrics import roc_auc_score, average_precision_score

OUT=ROOT/'reports/week5'
CATEGORIES=['bottle','screw','hazelnut','carpet','pill']
SUBSETS=[subset for n in (1,2,3) for subset in itertools.combinations(range(3),n)]


def write_csv(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    seed_all(0)
    OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'configs/week4.json').read_text())
    assert identity(config)==json.loads((ROOT/'results/week4/study.json').read_text())
    protocol=digest(identity(config))
    model=build_model(config,'cuda').eval()
    records=[]
    checks=[]
    stats=[]
    started=time.perf_counter()
    for category in CATEGORIES:
        dataset=MVTecDataset(ROOT/'data/mvtec_ad',category,'test',256)
        for seed in config['seeds']:
            folder=ROOT/'results/week4/runs'/category/f'seed_{seed}'
            receipt=json.loads((folder/'complete.json').read_text())
            assert receipt['protocol']==protocol
            for name in ('model.pt','maps.npz','scores.csv'):
                assert file_hash(folder/name)==receipt['files'][name]
            saved=torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
            restore_model(model,saved['trainable_state'])
            del saved
            parts=[[],[],[]]
            original=[]
            labels=[]
            masks=[]
            inference_start=time.perf_counter()
            with torch.inference_mode():
                for images,ys,truth,_ in DataLoader(dataset,batch_size=config['micro_batch_size']):
                    features=model._get_cnn_features(images.cuda())
                    hidden=[flow(feature)[0] for flow,feature in zip(model.fast_flow_blocks,features)]
                    levels=[torch.nn.functional.interpolate(-torch.exp(-.5*z.square().mean(1,keepdim=True)),
                        (256,256),mode='bilinear',align_corners=False) for z in hidden]
                    average=torch.stack(levels,dim=-1).mean(-1)
                    reference=model.anomaly_map_generator(hidden)
                    torch.testing.assert_close(average,reference,rtol=0,atol=0)
                    original.append(average[:,0].cpu().numpy())
                    for i,value in enumerate(levels):
                        parts[i].append(value[:,0].cpu().numpy())
                    labels.extend(ys.tolist())
                    masks.append(truth.numpy())
            levels=[np.concatenate(part) for part in parts]
            original=np.concatenate(original)
            labels=np.array(labels)
            masks=np.concatenate(masks)
            with np.load(folder/'maps.npz') as archive:
                baseline=archive['maps']
            np.testing.assert_allclose(original,baseline,rtol=1e-5,atol=1e-6)
            checks.append({'category':category,'seed':seed,'images':len(dataset),
                'component_average_matches_generator_exactly':True,
                'inference_and_assembly_seconds':time.perf_counter()-inference_start,
                'max_saved_map_difference':float(np.abs(original-baseline).max())})
            for i,values in enumerate(levels):
                stats.append({'category':category,'seed':seed,'level':i+1,
                    'test_normal_mean':float(values[labels==0].mean()),'test_normal_std':float(values[labels==0].std()),
                    'test_normal_min':float(values[labels==0].min()),'test_normal_max':float(values[labels==0].max()),
                    'note':'Descriptive test statistics only. Not fitted calibration.'})
            metric_start=time.perf_counter()
            for subset in SUBSETS:
                values=original if len(subset)==3 else np.mean([levels[i] for i in subset],axis=0)
                assert np.isfinite(values).all()
                scores=values.max((1,2))
                y,x=np.unravel_index(values.reshape(len(values),-1).argmax(1),(256,256))
                border=(y<16)|(y>=240)|(x<16)|(x>=240)
                inside=masks[np.arange(len(masks)),y,x].astype(bool)
                records.append({'category':category,'seed':seed,'levels':'+'.join(str(i+1) for i in subset),
                    'image_auroc':float(roc_auc_score(labels,scores)),
                    'pixel_auroc':float(roc_auc_score(masks.ravel(),values.ravel())),
                    'pixel_ap':float(average_precision_score(masks.ravel(),values.ravel())),
                    'normal_peak_border_fraction':float(border[labels==0].mean()),
                    'defect_peak_inside_mask_fraction':float(inside[labels==1].mean())})
            checks[-1]['seven_subset_metric_seconds']=time.perf_counter()-metric_start
            write_csv(OUT/'level_subsets.csv',records)
            write_csv(OUT/'level_ranges.csv',stats)
            write_json(OUT/'level_checks.json',{'complete':False,'runs':checks,'protocol':protocol})
            print(f'Level comparisons: {len(checks)}/15, {category} seed {seed}',flush=True)
    write_json(OUT/'level_checks.json',{'complete':True,'runs':checks,'protocol':protocol,
        'elapsed_seconds':time.perf_counter()-started,'source_sha256':file_hash(__file__)})
    lines=['# Looking at the three feature levels','','These are inference-only output ablations of the saved Week 4 models. The three-level average is the original baseline. No model was trained or selected using these scores. Every category uses all three seeds. These five categories were selected after inspecting the earlier test results, so this is exploratory.', '',
        '| Category | Levels | Image AUROC | Pixel AUROC | Pixel AP |','| --- | --- | ---: | ---: | ---: |']
    for category in CATEGORIES:
        for subset in SUBSETS:
            label='+'.join(str(i+1) for i in subset)
            selected=[r for r in records if r['category']==category and r['levels']==label]
            values=[f"{100*statistics.mean(r[k] for r in selected):.2f} ± {100*statistics.stdev(r[k] for r in selected):.2f}%"
                for k in ('image_auroc','pixel_auroc','pixel_ap')]
            lines.append(f'| {category} | {label} | '+ ' | '.join(values)+' |')
    lines += ['','Levels 1, 2 and 3 have native sizes 64 x 64, 32 x 32 and 16 x 16. Subsets use equal averaging after bilinear resizing to 256 x 256. Image scoring always uses the maximum. Standard deviations are across the three seeds. These are output ablations of jointly trained independent branches, not independently trained single-level models.', '',
        'All 15 extracted three-level averages matched the model map generator exactly and passed comparison with the saved maps (rtol 1e-5, atol 1e-6). [Exact differences](level_checks.json) and [all seed results](level_subsets.csv) are saved. [Level ranges](level_ranges.csv) describe normal test maps and must not be used as training calibration.', '',
        'Do not pick a separate best subset for each category from this table. The feature-fusion pilot uses the design fixed in the [Week 5 plan](../../docs/WEEK5_PLAN.md).', '']
    (OUT/'levels.md').write_text('\n'.join(lines),encoding='utf-8')


if __name__=='__main__':
    main()
