"""Audit the completed comparison and make paired heatmaps for the midterm."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import csv
import hashlib
import html
import json
from pathlib import Path
import statistics
import numpy as np
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/week5/analysis'
GALLERY = ROOT / 'results/week5/comparison'
METRICS = ['image_auroc', 'pixel_auroc', 'pixel_average_precision']
ARMS = ['middle', 'fusion']


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def csvwrite(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def summarize_images():
    with (OUT/'per_image.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    areas = []
    for low, high, label in [(0,.001,'up to 0.1%'),(.001,.01,'0.1% to 1%'),(.01,.05,'1% to 5%'),(.05,1,'over 5%')]:
        entry = {'defect_area':label}
        for arm in ARMS:
            subset = [r for r in rows if r['arm']==arm and r['label']=='1' and low<float(r['mask_fraction'])<=high]
            entry['images'] = len(subset)//3
            entry[arm+'_mean_per_image_ap'] = statistics.mean(float(r['pixel_ap']) for r in subset)
        areas.append(entry)
    csvwrite(OUT/'defect_area.csv',areas)
    peaks = []
    for arm in ARMS:
        subset = [r for r in rows if r['arm']==arm and r['label']=='1']
        peaks.append({'arm':arm,'defective_images':len(subset)//3,
                      'peak_inside_mask_fraction':statistics.mean(r['peak_inside_mask']=='True' for r in subset),
                      'peak_in_16px_border_fraction':statistics.mean(r['peak_on_border']=='True' for r in subset)})
    csvwrite(OUT/'peak_locations.csv',peaks)


def preview(category, index, rows, masks, maps, reason):
    row = rows[index]
    source = ROOT / 'data/mvtec_ad' / category / row['image']
    original = Image.open(source).convert('RGB').resize((256, 256))
    mask = Image.fromarray(masks[index]*255).convert('RGB')
    values = [maps[(seed, arm)][index] for seed in range(3) for arm in ARMS]
    # Same numeric range across all six maps for this image, not per-model scaling.
    low, high = min(float(v.min()) for v in values), max(float(v.max()) for v in values)
    canvas = Image.new('RGB', (1024, 600), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), f'{category}: {row["image"]} | {reason}', fill='black')
    draw.text((8, 22), f'Shared map range for this image: {low:.5g} to {high:.5g}. Yellow = higher anomaly score.', fill='black')
    panels = [('Original', original), ('Defect mask', mask)]
    for seed in range(3):
        for arm in ARMS:
            value = maps[(seed, arm)][index]
            scaled = (value-low)/max(high-low, 1e-12)
            heat = Image.fromarray((plt.colormaps['inferno'](scaled)[...,:3]*255).astype(np.uint8))
            panels.append((f'{arm}, seed {seed}', Image.blend(original, heat, .55)))
    for i, (title, panel) in enumerate(panels):
        x, y = (i%4)*256, 45+(i//4)*278
        draw.text((x+6, y), title, fill='black')
        canvas.paste(panel, (x, y+18))
    name = f'{category}_{reason}_{Path(row["image"]).parent.name}_{Path(row["image"]).stem}.jpg'
    canvas.save(GALLERY/name, quality=90)
    return name


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    GALLERY.mkdir(parents=True, exist_ok=True)
    store = ROOT / 'results/week5/study'
    study = read(store/'study.json')
    for name, expected in study['sources'].items():
        assert sha(ROOT/name)==expected, name
    complete = read(store/'summary.json')
    assert complete['complete'] and len(complete['runs'])==90
    runrows, image_rows, defectrows, histories, selected = [], [], [], [], []
    verified = 0
    for category in study['config']['categories']:
        projection = read(store/category/'projection.json')
        assert sha(store/category/'projection.pt')==projection['sha256']
        maps, per_image_ap, base_rows = {}, {}, None
        masks = None
        for seed in range(3):
            initial_hash = None
            for arm in ARMS:
                folder = store/category/f'seed_{seed}'/arm
                receipt = read(folder/'complete.json')
                assert receipt['protocol']==complete['protocol']
                for name, expected in receipt['files'].items():
                    assert sha(folder/name)==expected, str(folder/name)
                metrics = read(folder/'metrics.json')
                assert metrics['checkpoint_reload_exact'] and metrics['backbone_unchanged']
                if initial_hash is None:
                    initial_hash = metrics['initial_state_hash']
                assert initial_hash==metrics['initial_state_hash']
                with (folder/'scores.csv').open(newline='') as handle:
                    rows = list(csv.DictReader(handle))
                if base_rows is None:
                    base_rows = rows
                    masks_list = []
                    for row in rows:
                        p = Path(row['image'])
                        if int(row['label']):
                            mask = Image.open(ROOT/'data/mvtec_ad'/category/'ground_truth'/p.parent.name/(p.stem+'_mask.png')).convert('L')
                            masks_list.append((np.asarray(mask.resize((256,256), Image.Resampling.NEAREST))>0).astype(np.uint8))
                        else:
                            masks_list.append(np.zeros((256,256),dtype=np.uint8))
                    masks = np.stack(masks_list)
                assert [(r['image'],r['label']) for r in rows]==[(r['image'],r['label']) for r in base_rows]
                with np.load(folder/'maps.npz', allow_pickle=False) as saved:
                    values = saved['maps']
                assert values.shape==masks.shape and np.isfinite(values).all()
                maps[(seed,arm)] = values
                labels = np.array([int(r['label']) for r in rows])
                scores = values.reshape(len(rows),-1).max(axis=1)
                np.testing.assert_array_equal(scores, np.array([float(r['score']) for r in rows]))
                measured = [roc_auc_score(labels,scores), roc_auc_score(masks.ravel(),values.ravel()), average_precision_score(masks.ravel(),values.ravel())]
                for key,value in zip(METRICS,measured):
                    assert abs(value-metrics[key])<1e-12, (category,seed,arm,key)
                runrows.append({'category':category,'seed':seed,'arm':arm, **{k:metrics[k] for k in METRICS},
                                'zero_fraction':float(np.mean(values==0)), 'unique_image_scores':len(np.unique(scores)), 'test_images':len(rows)})
                for defect, item in metrics['defects'].items():
                    subset=np.array([r['defect'] in ('good',defect) for r in rows])
                    auc=float(roc_auc_score(labels[subset],scores[subset]))
                    assert abs(auc-item['image_auroc'])<1e-12
                    defectrows.append({'category':category,'seed':seed,'arm':arm,'defect':defect,'image_auroc':auc})
                aps=[]
                for i,(value,mask,row) in enumerate(zip(values,masks,rows)):
                    positive=mask.astype(bool)
                    ap=float(average_precision_score(mask.ravel(),value.ravel())) if positive.any() else None
                    aps.append(ap if ap is not None else np.nan)
                    y,x=np.unravel_index(value.argmax(),value.shape)
                    image_rows.append({'category':category,'seed':seed,'arm':arm,'image':row['image'],'label':row['label'],
                                       'defect':row['defect'],'pixel_ap':ap,'mask_fraction':float(mask.mean()),
                                       'peak_inside_mask':bool(mask[y,x]),'peak_on_border':bool(min(y,x,255-y,255-x)<16),
                                       'defect_mean':float(value[positive].mean()) if positive.any() else None,
                                       'outside_mean':float(value[~positive].mean()),'score':float(scores[i])})
                per_image_ap[(seed,arm)]=np.array(aps)
                history=read(folder/'training.json')
                assert len(history)==500 and [r['epoch'] for r in history]==list(range(1,501))
                assert all(np.isfinite(r[k]) for r in history for k in ('loss','max_gradient_norm','gaussian_term','log_jacobian'))
                losses=np.array([r['loss'] for r in history])
                histories.append({'category':category,'seed':seed,'arm':arm,'final_loss':float(losses[-1]),
                                  'final50_mean_loss':float(losses[-50:].mean()),'max_upward_loss_step':float(np.diff(losses).max()),
                                  'largest_upward_step_epoch':int(np.argmax(np.diff(losses))+2),
                                  'last25_max_upward_step':float(np.diff(losses[-26:]).max()),
                                  'final_gradient_norm':history[-1]['max_gradient_norm'],
                                  'active_training_seconds_excluding_steps_over_60s':sum(r['seconds'] for r in history if r['seconds']<60)})
                verified+=1
                print(f'Checked {verified}/90: {category}, seed {seed}, {arm}',flush=True)
        delta=np.mean([per_image_ap[(s,'fusion')]-per_image_ap[(s,'middle')] for s in range(3)],axis=0)
        abnormal=np.flatnonzero(np.array([int(r['label']) for r in rows])==1)
        normal=np.flatnonzero(np.array([int(r['label']) for r in rows])==0)
        # Fixed selection rule includes relative gains, regressions and a difficult normal image.
        ranks=[]
        for s in range(3):
            scores=maps[(s,'fusion')].reshape(len(rows),-1).max(axis=1)
            ranks.append(np.searchsorted(np.sort(scores),scores,side='right')/len(scores))
        worst_normal=normal[np.argmax(np.mean(ranks,axis=0)[normal])]
        choices=[('largest_AP_gain',abnormal[np.argmax(delta[abnormal])]),('largest_AP_drop',abnormal[np.argmin(delta[abnormal])]),('high_rank_normal',worst_normal)]
        for reason,index in choices:
            name=preview(category,index,rows,masks,maps,reason)
            selected.append({'category':category,'reason':reason,'image':rows[index]['image'],'mean_AP_change_points':float(delta[index]*100) if np.isfinite(delta[index]) else None,'preview':name})
        del maps,masks
    categoryrows=[]
    overall={}
    for metric in METRICS:
        overall[metric]={}
        for arm in ARMS:
            seedmeans=[statistics.mean(r[metric] for r in runrows if r['arm']==arm and r['seed']==s) for s in range(3)]
            overall[metric][arm]={'mean':statistics.mean(seedmeans),'std':statistics.stdev(seedmeans),'seed_means':seedmeans}
        differences=np.array(overall[metric]['fusion']['seed_means'])-overall[metric]['middle']['seed_means']
        overall[metric]['paired_seed_changes_points']=(100*differences).tolist()
    for category in study['config']['categories']:
        entry={'category':category}
        for metric in METRICS:
            for arm in ARMS:
                vals=[r[metric] for r in runrows if r['category']==category and r['arm']==arm]
                entry[f'{arm}_{metric}']=statistics.mean(vals)
                entry[f'{arm}_{metric}_std']=statistics.stdev(vals)
            entry[f'delta_{metric}_points']=100*(entry[f'fusion_{metric}']-entry[f'middle_{metric}'])
            entry[f'{metric}_paired_seed_wins']=sum(next(r[metric] for r in runrows if r['category']==category and r['arm']=='fusion' and r['seed']==s)>next(r[metric] for r in runrows if r['category']==category and r['arm']=='middle' and r['seed']==s) for s in range(3))
        categoryrows.append(entry)
    for name,rows in [('per_run',runrows),('per_category',categoryrows),('per_image',image_rows),('per_defect',defectrows),('training',histories),('selected',selected)]:
        csvwrite(OUT/f'{name}.csv',rows)
    summarize_images()
    result={'verified_runs':verified,'overall':overall,'source_sha256':sha(Path(__file__)),'protocol':complete['protocol'],
            'all_metrics_recomputed':True,'all_receipts_verified':True,'all_histories_finite':True,
            'projection_hashes_verified':len(study['config']['categories']),
            'paired_initializations_verified':verified//2,
            'recorded_reload_and_backbone_checks_passed':verified}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(14,6),layout='constrained')
    for ax,metric,title in zip(axes,METRICS,['Image AUROC','Pixel AUROC','Pixel AP']):
        vals=[r[f'delta_{metric}_points'] for r in categoryrows]
        ax.barh([r['category'] for r in categoryrows],vals,color=['#298c55' if v>0 else '#be493d' for v in vals])
        ax.axvline(0,color='black',linewidth=.7)
        ax.set_title(title)
        ax.set_xlabel('Fusion minus middle control (percentage points)')
    fig.savefig(OUT/'category_changes.png',dpi=160)
    plt.close(fig)
    blocks=['<!doctype html><meta charset="utf-8"><title>Week 5 paired heatmaps</title><style>body{font:16px Arial;margin:24px;background:#eee}img{max-width:100%;}article{background:white;padding:16px;margin:20px 0}</style>',
            '<h1>Week 5 paired heatmaps</h1><p>Same images and masks, both models, all three seeds. Each preview uses one shared numeric color range across its six maps. Colors cannot be compared across different previews. These are selected diagnostic examples, not a random sample or independent validation.</p>',
            '<p>For every category: largest and smallest mean per-image pixel AP change among defective images, and the normal image with highest average fusion score rank. A relative gain may still be poor localization.</p>']
    for row in selected:
        blocks.append(f'<article><h2>{html.escape(row["category"]+": "+row["reason"])}</h2><p>{html.escape(row["image"])} | AP change: {row["mean_AP_change_points"]}</p><img loading="lazy" src="{row["preview"]}"></article>')
    (GALLERY/'index.html').write_text('\n'.join(blocks),encoding='utf-8')
    print(json.dumps(overall,indent=2),flush=True)


if __name__=='__main__':
    main()
