"""Compare the completed studies and export the same images for each model."""
import os
os.environ['MPLBACKEND'] = 'Agg'
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import csv
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/comparison'
MODELS = ['week3', 'week4', 'middle', 'fusion']
NAMES = ['Week 3: ResNet-18', 'Week 4: WideResNet', 'Week 5: middle', 'Week 5: fusion']
METRICS = ['image_auroc', 'pixel_auroc', 'pixel_average_precision']
TEXTURES = {'carpet', 'grid', 'leather', 'tile', 'wood'}
EXTRA = {'bottle': ['test/contamination/005.png'], 'carpet': ['test/hole/004.png'],
         'hazelnut': ['test/hole/007.png'], 'pill': ['test/contamination/018.png'],
         'screw': ['test/scratch_head/010.png'], 'wood': ['test/hole/008.png']}


def read(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path, values):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=values[0].keys())
        writer.writeheader()
        writer.writerows(values)


def folder(model, category, seed):
    if model=='week3':
        return ROOT/'results/week3'/category/f'seed_{seed}'
    if model=='week4':
        return ROOT/'results/week4/runs'/category/f'seed_{seed}'
    return ROOT/'results/week5/study'/category/f'seed_{seed}'/model


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def font(size):
    try:
        return ImageFont.truetype('arial.ttf', size)
    except OSError:
        return ImageFont.load_default(size=size)


def timing_summary():
    summary={}
    for week,pattern in [('week4','results/week4/runs/*/seed_*/training.json'),
                         ('week5','results/week5/study/*/seed_*/*/training.json')]:
        records=[]
        for path in sorted(ROOT.glob(pattern)):
            history=read(path)
            assert len(history)==500
            records.append({'history':path.relative_to(ROOT).as_posix(),
                            'sha256':digest(path),
                            'median_seconds_after_first':statistics.median(r['seconds'] for r in history[1:])})
        summary[week]={'runs':len(records),'estimated_training_hours':sum(r['median_seconds_after_first']*500 for r in records)/3600,'records':records}
    (OUT/'timing.json').write_text(json.dumps(summary,indent=2)+'\n')


def panel(category, image_path, original, mask, maps, models, destination):
    size = 200
    width = (2+len(models))*size
    canvas = Image.new('RGB', (width, 800), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8,8), f'{category}: {image_path}', fill='black', font=font(18))
    draw.text((8,34), 'Each map uses its own range. Compare locations and AP, not brightness between models.', fill='black', font=font(13))
    base = original.resize((size,size))
    gt = Image.fromarray(mask*255).convert('RGB').resize((size,size), Image.Resampling.NEAREST)
    for seed in range(3):
        y = 68+seed*240
        for col,(name,img) in enumerate([('Original',base),('Defect mask',gt)]):
            draw.text((col*size+5,y),name,fill='black',font=font(14))
            canvas.paste(img,(col*size,y+30))
        for col,model in enumerate(models,2):
            value=maps[(model,seed)]
            scaled=(value-value.min())/max(float(value.max()-value.min()),1e-12)
            heat=Image.fromarray((plt.colormaps['inferno'](scaled)[...,:3]*255).astype(np.uint8))
            overlay=Image.blend(original,heat,.55).resize((size,size))
            marker=ImageDraw.Draw(overlay)
            py,px=np.unravel_index(value.argmax(),value.shape)
            px,py=int(px*size/256),int(py*size/256)
            marker.line((px-4,py,px+4,py),fill='cyan',width=2)
            marker.line((px,py-4,px,py+4),fill='cyan',width=2)
            ap=average_precision_score(mask.ravel(),value.ravel())*100 if mask.any() else None
            title=f'{NAMES[MODELS.index(model)]}, s{seed}'
            draw.text((col*size+4,y-3),title,fill='black',font=font(13))
            draw.text((col*size+4,y+12),f'Image pixel AP: {ap:.1f}%' if ap is not None else 'Normal image',fill='black',font=font(12))
            canvas.paste(overlay,(col*size,y+30))
    canvas.save(destination,quality=88)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'heatmaps').mkdir(exist_ok=True)
    for week in ('week3','week4','week5'):
        (ROOT/f'reports/{week}/heatmaps').mkdir(exist_ok=True)
    categories=read(ROOT/'configs/week5_study.json')['categories']
    all_runs=[]
    manifest=[]
    selections=[]
    image_stats=[]
    for category in categories:
        reference=rows(folder('week3',category,0)/'scores.csv')
        for r in reference:
            r['image']=r['image'].replace('\\','/')
        identifiers=[r['image'] for r in reference]
        normal=next(r['image'] for r in sorted(reference,key=lambda r:r['image']) if int(r['label'])==0)
        defective=next(r['image'] for r in sorted(reference,key=lambda r:r['image']) if int(r['label'])==1)
        chosen=list(dict.fromkeys([normal,defective]+EXTRA.get(category,[])))
        selected_maps={im:{} for im in chosen}
        for model in MODELS:
            for seed in range(3):
                f=folder(model,category,seed)
                scores=rows(f/'scores.csv')
                assert [(r['image'].replace('\\','/'),r['label']) for r in scores]==[(r['image'],r['label']) for r in reference]
                metrics=read(f/'metrics.json')
                if model!='week3':
                    receipt=read(f/'complete.json')
                    for name in ('metrics.json','scores.csv','maps.npz'):
                        assert digest(f/name)==receipt['files'][name]
                with np.load(f/'maps.npz',allow_pickle=False) as saved:
                    values=saved['maps']
                assert np.isfinite(values).all() and values.shape==(len(scores),256,256)
                computed=values.reshape(len(scores),-1).max(axis=1)
                np.testing.assert_array_equal(computed,np.array([float(r['score']) for r in scores]))
                auc=roc_auc_score([int(r['label']) for r in scores],computed)
                assert abs(auc-metrics['image_auroc'])<1e-12
                all_runs.append({'model':model,'category':category,'seed':seed,**{k:metrics[k] for k in METRICS}})
                manifest.append({'model':model,'category':category,'seed':seed,
                                 **{name.replace('.','_')+'_sha256':digest(f/name) for name in ('metrics.json','scores.csv','maps.npz')}})
                for im in chosen:
                    selected_maps[im][(model,seed)]=values[identifiers.index(im)].copy()
                del values
        for im in chosen:
            source=ROOT/'data/mvtec_ad'/category/im
            original=Image.open(source).convert('RGB').resize((256,256))
            rel=Path(im)
            mask=np.zeros((256,256),dtype=np.uint8)
            if rel.parent.name!='good':
                gt=ROOT/'data/mvtec_ad'/category/'ground_truth'/rel.parent.name/(rel.stem+'_mask.png')
                mask=(np.asarray(Image.open(gt).convert('L').resize((256,256),Image.Resampling.NEAREST))>0).astype(np.uint8)
            name=f'{category}_{rel.parent.name}_{rel.stem}.jpg'
            reason='First normal image by filename' if im==normal else ('First defective image by filename' if im==defective else 'Previously inspected diagnostic example')
            selections.append({'category':category,'image':im,'selection':reason,'file':name})
            panel(category,im,original,mask,selected_maps[im],MODELS,OUT/'heatmaps'/name)
            for week,models in [('week3',['week3']),('week4',['week4']),('week5',['middle','fusion'])]:
                panel(category,im,original,mask,selected_maps[im],models,ROOT/f'reports/{week}/heatmaps'/name)
            for model in MODELS:
                for seed in range(3):
                    v=selected_maps[im][(model,seed)]
                    image_stats.append({'category':category,'image':im,'model':model,'seed':seed,
                        'pixel_ap':float(average_precision_score(mask.ravel(),v.ravel())) if mask.any() else '',
                        'map_min':float(v.min()),'map_max':float(v.max())})
        print(f'Compared {category}: 12 runs, {len(chosen)} matched images',flush=True)
    save_csv(OUT/'per_run.csv',all_runs)
    save_csv(OUT/'source_files.csv',manifest)
    save_csv(OUT/'selected_images.csv',selections)
    save_csv(OUT/'selected_image_metrics.csv',image_stats)
    overall={}
    catrows=[]
    for model in MODELS:
        overall[model]={}
        for metric in METRICS:
            vals=[statistics.mean(r[metric] for r in all_runs if r['model']==model and r['seed']==s) for s in range(3)]
            overall[model][metric]={'mean':statistics.mean(vals),'std':statistics.stdev(vals),'seed_means':vals}
        for c in categories:
            subset=[r for r in all_runs if r['model']==model and r['category']==c]
            catrows.append({'model':model,'category':c,**{k:statistics.mean(r[k] for r in subset) for k in METRICS}})
    save_csv(OUT/'per_category.csv',catrows)
    paired=[]
    for c in categories:
        for metric in METRICS:
            deltas=[next(r[metric] for r in all_runs if r['model']=='fusion' and r['category']==c and r['seed']==s)-next(r[metric] for r in all_runs if r['model']=='middle' and r['category']==c and r['seed']==s) for s in range(3)]
            paired.append({'category':c,'metric':metric,'mean_change_points':100*statistics.mean(deltas),'seed_0':100*deltas[0],'seed_1':100*deltas[1],'seed_2':100*deltas[2]})
    save_csv(OUT/'paired_changes.csv',paired)
    sensitivity=[]
    for metric in METRICS:
        changes={r['category']:r['mean_change_points'] for r in paired if r['metric']==metric}
        for group,cats in [('all',categories),('textures',[c for c in categories if c in TEXTURES]),('objects',[c for c in categories if c not in TEXTURES])]:
            sensitivity.append({'metric':metric,'group':group,'categories':len(cats),'mean_change_points':statistics.mean(changes[c] for c in cats),'median_category_change_points':statistics.median(changes[c] for c in cats)})
        for excluded in categories:
            sensitivity.append({'metric':metric,'group':'excluding_'+excluded,'categories':14,'mean_change_points':statistics.mean(v for c,v in changes.items() if c!=excluded),'median_category_change_points':statistics.median(v for c,v in changes.items() if c!=excluded)})
    save_csv(OUT/'sensitivity.csv',sensitivity)
    timing_summary()
    (OUT/'summary.json').write_text(json.dumps({'overall':overall,'runs':len(all_runs),'selected_images':len(selections),'analysis_source_sha256':digest(Path(__file__)),'image_metrics_recomputed':True,'pixel_metrics_source':'Recorded metrics, previously audited from saved maps in each weekly analysis.'},indent=2)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,metric,title in zip(axes,METRICS,['Image AUROC','Pixel AUROC','Pixel AP']):
        ax.bar(range(4),[overall[m][metric]['mean']*100 for m in MODELS],yerr=[overall[m][metric]['std']*100 for m in MODELS],capsize=4,color=['#5877a8','#2b8c66','#a18b62','#b45c57'])
        ax.set_xticks(range(4),['W3','W4','W5 middle','W5 fusion'],rotation=15)
        ax.set_ylim(0,105);ax.set_title(title);ax.set_ylabel('%')
    fig.savefig(OUT/'overview.png',dpi=160);plt.close(fig)
    for week in ['comparison','week3','week4','week5']:
        target=ROOT/f'reports/{week}/HEATMAPS.md'
        intro='# Heatmap examples\n\n[Back to the overview](README.md)\n\nThese are saved test predictions, shown for seeds 0, 1 and 2. The cyan cross marks the highest-scoring pixel. Each map has its own display range, so compare locations and the printed per-image pixel AP, not brightness across models. Normal images have no defect mask and no per-image AP.\n\nEvery category includes its first normal and first defective test image in filename order. Six previously inspected examples are added to connect the gallery to the discussion. This is a disclosed selection, not a representative sample or a new test set. Full category metrics are the main evidence. Images are from [MVTec AD, CC BY-NC-SA 4.0](https://www.mvtec.com/research-teaching/datasets/mvtec-ad).\n'
        for row in selections:
            intro+=f"\n## {row['category']}: {row['image']}\n\n{row['selection']}.\n\n![{row['category']} {row['image']}](heatmaps/{row['file']})\n"
        target.write_text(intro,encoding='utf-8')
    print('Saved four shareable galleries and the comparison tables.',flush=True)


if __name__=='__main__':
    main()
