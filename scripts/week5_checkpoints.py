"""Evaluate fixed late checkpoints without retraining or changing study results."""
import csv
import gc
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.week5_model import ProjectedFlow, Projection, seed_all, torch, np
from week3 import build_model, MVTecDataset, DataLoader, file_hash
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = ROOT/'reports/week5/checkpoints'
CATEGORIES = ['bottle','carpet','hazelnut','screw']
EPOCHS = [400,450,475,500]


def read(path):
    return json.loads(path.read_text())


def summarize():
    import statistics
    import matplotlib.pyplot as plt
    with (OUT/'per_run.csv').open(newline='') as f:
        records=list(csv.DictReader(f))
    table=[]
    fig,axes=plt.subplots(4,2,figsize=(11,11),layout='constrained')
    for i,category in enumerate(CATEGORIES):
        for arm,color in [('middle','#3976ac'),('fusion','#c56b36')]:
            for epoch in EPOCHS:
                subset=[r for r in records if r['category']==category and r['arm']==arm and int(r['epoch'])==epoch]
                assert len(subset)==3
                table.append({'category':category,'arm':arm,'epoch':epoch,
                              **{m:statistics.mean(float(r[m]) for r in subset) for m in ['image_auroc','pixel_auroc','pixel_average_precision']}})
            for j,metric in enumerate(['image_auroc','pixel_average_precision']):
                for seed in range(3):
                    vals=[float(next(r[metric] for r in records if r['category']==category and r['arm']==arm and int(r['epoch'])==e and int(r['seed'])==seed))*100 for e in EPOCHS]
                    axes[i,j].plot(EPOCHS,vals,color=color,alpha=.18,linewidth=1)
                vals=[next(r[metric] for r in table if r['category']==category and r['arm']==arm and r['epoch']==e)*100 for e in EPOCHS]
                axes[i,j].plot(EPOCHS,vals,'o-',color=color,label=arm,linewidth=2)
                axes[i,j].set_title(category+': '+('Image AUROC' if j==0 else 'Pixel AP'))
                axes[i,j].set_xticks(EPOCHS);axes[i,j].set_ylim(0,100);axes[i,j].set_ylabel('%')
        axes[i,0].legend()
    fig.savefig(OUT/'trajectories.png',dpi=140);plt.close(fig)
    with (OUT/'means.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    store=ROOT/'results/week5/study'
    study=read(store/'study.json')
    for name,expected in study['sources'].items():
        assert file_hash(ROOT/name)==expected
    config=study['config']
    seed_all(0)
    original=build_model(read(ROOT/'configs/week4.json'),'cuda')
    backbone=original.feature_extractor
    del original
    backbone.eval().requires_grad_(False)
    output=[]
    checks=[]
    for category in CATEGORIES:
        projection_path=store/category/'projection.pt'
        projection_sha=file_hash(projection_path)
        assert projection_sha==read(store/category/'projection.json')['sha256']
        projection=Projection(torch.load(projection_path,map_location='cpu',weights_only=True),config['grid']).cuda().eval()
        dataset=MVTecDataset(ROOT/'data/mvtec_ad',category,'test',256)
        features={a:[] for a in config['arms']}
        labels=[];masks=[];paths=[]
        with torch.inference_mode():
            for images,ys,ms,ps in DataLoader(dataset,batch_size=4):
                native=backbone(images.cuda())
                for arm in config['arms']:
                    features[arm].append(projection(native,arm).cpu())
                labels.extend(ys.tolist());masks.extend(ms.numpy());paths.extend(ps)
        masks=np.stack(masks)
        for arm in config['arms']:
            features[arm]=torch.cat(features[arm])
        for seed in config['seeds']:
            for arm in config['arms']:
                base=store/category/f'seed_{seed}'/arm
                receipt=read(base/'complete.json')
                for epoch in EPOCHS:
                    path=base/f'model_epoch_{epoch:04d}.pt'
                    assert file_hash(path)==receipt['files'][path.name]
                    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
                    assert checkpoint['projection_sha256']==projection_sha and checkpoint['epoch']==epoch
                    assert checkpoint['config']==config and checkpoint['protocol']==receipt['protocol']
                    model=ProjectedFlow(config).cuda().eval()
                    model.load_state_dict(checkpoint['model'],strict=True)
                    maps=[]
                    with torch.inference_mode():
                        for start in range(0,len(dataset),4):
                            maps.extend(model.anomaly_map(features[arm][start:start+4].cuda()).cpu().numpy()[:,0])
                    maps=np.stack(maps)
                    assert np.isfinite(maps).all()
                    row={'category':category,'seed':seed,'arm':arm,'epoch':epoch,
                         'image_auroc':float(roc_auc_score(labels,maps.reshape(len(dataset),-1).max(1))),
                         'pixel_auroc':float(roc_auc_score(masks.ravel(),maps.ravel())),
                         'pixel_average_precision':float(average_precision_score(masks.ravel(),maps.ravel()))}
                    if epoch==500:
                        assert file_hash(base/'maps.npz')==receipt['files']['maps.npz']
                        with np.load(base/'maps.npz',allow_pickle=False) as saved:
                            reference=saved['maps']
                        np.testing.assert_allclose(maps,reference,rtol=1e-5,atol=1e-6)
                        recorded=read(base/'metrics.json')
                        for metric in ['image_auroc','pixel_auroc','pixel_average_precision']:
                            assert abs(recorded[metric]-row[metric])<1e-6
                        checks.append({'category':category,'seed':seed,'arm':arm,'max_absolute_map_difference':float(np.abs(maps-reference).max()),'exact_map_match':bool(np.array_equal(maps,reference))})
                    output.append(row)
                    with (OUT/'per_run.csv').open('w',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=output[0]);writer.writeheader();writer.writerows(output)
                    print(f"{category} seed {seed} {arm} epoch {epoch}: image {row['image_auroc']*100:.2f}, pixel AP {row['pixel_average_precision']*100:.2f}",flush=True)
                    del model,maps
                gc.collect();torch.cuda.empty_cache()
        del features,projection
        gc.collect();torch.cuda.empty_cache()
    (OUT/'checks.json').write_text(json.dumps({'categories':CATEGORIES,'epochs':EPOCHS,'evaluations':len(output),'final_replays':checks,'source_sha256':file_hash(__file__),'selection':'Fixed diagnostic subset after inspecting final test results. All retained checkpoints reported, no replacement of final results.'},indent=2)+'\n')
    summarize()


if __name__=='__main__':
    main()
