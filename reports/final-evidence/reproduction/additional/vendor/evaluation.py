"""Common fixed-endpoint metrics. This module is never invoked on real test data by recovery tests."""
from common import *
import numpy as np
from PIL import Image,ImageDraw
from sklearn.metrics import roc_auc_score,average_precision_score

def load_map(item):
    if sha(item['map_file'])!=item['map_sha256']:raise RuntimeError('Prediction checksum mismatch')
    return np.load(item['map_file'],allow_pickle=False)
def mask(row,size):
    if not row['label']:return np.zeros((size,size),np.uint8)
    with Image.open(row['mask']) as im:return (np.asarray(im.convert('L').resize((size,size),Image.Resampling.NEAREST))>0).astype(np.uint8)
def exact_metrics(scores,labels,maps,masks):
    return {'image_auroc':float(roc_auc_score(labels,scores)),
        'pixel_auroc':float(roc_auc_score(masks.reshape(-1),maps.reshape(-1))),
        'pixel_ap':float(average_precision_score(masks.reshape(-1),maps.reshape(-1)))}
def gallery(rows,pred,folder):
    dest=folder/'gallery';dest.mkdir(exist_ok=True)
    groups=sorted({(r.get('condition',''),r['defect']) for r in rows})
    selected=[min((r for r in rows if (r.get('condition',''),r['defect'])==g),key=lambda r:r['path']) for g in groups]
    for row in selected:
        guard();a=load_map(pred[row['path']]);m=mask(row,256)
        with Image.open(row['path']) as im:rgb=im.convert('RGB').resize((256,256))
        v=(a-float(a.min()))/max(float(np.ptp(a)),1e-12)
        heat=Image.fromarray((np.stack([v,.5*np.sqrt(v),1-v],-1)*255).astype(np.uint8))
        canvas=Image.new('RGB',(768,292),'white');draw=ImageDraw.Draw(canvas)
        for i,(picture,title) in enumerate([(rgb,'Image'),(Image.fromarray(m*255).convert('RGB'),'Released mask'),(Image.blend(rgb,heat,.5),'Overlay: per-image display scale')]):
            canvas.paste(picture,(i*256,30));draw.text((i*256+3,12),title,fill='black')
        canvas.save(dest/(key(row)+'.png'))
    write(dest/'selection.json',{'rule':'First absolute filename per condition/defect group, independent of predictions','paths':[r['path'] for r in selected],
        'display':'Per-image min/max for visibility only; compare numerical scores/metrics, not colors across sheets. Zero mask means nondefect, not physical background.'})
def mvtec(source,folder):
    rows=source['evaluation'];index={r['path']:r for r in read(folder/'index.json')}
    needed=len(rows)*256*256*40+2**30
    if resources()['ram_available']<needed:raise RuntimeError('Insufficient RAM for exact category metrics; predictions retained')
    maps=np.stack([load_map(index[r['path']]) for r in rows]);masks=np.stack([mask(r,256) for r in rows])
    result=exact_metrics([index[r['path']]['score'] for r in rows],[r['label'] for r in rows],maps,masks)
    result.update(images=len(rows),metric_grid=[256,256],mask_resize='nearest',threshold_tuning=False)
    result['per_defect_image_auroc']={d:float(roc_auc_score([r['label'] for r in rows if r['defect'] in ('good',d)],
        [index[r['path']]['score'] for r in rows if r['defect'] in ('good',d)])) for d in sorted({r['defect'] for r in rows}-{'good'})}
    del maps,masks
    write(folder/'metrics.json',result);gallery(rows,index,folder)
def aebad(source,folder):
    import cv2
    from scipy.ndimage import label
    sys.path.insert(0,str(REPO/'scripts'))
    from paired_lighting_metrics import RegionMetrics
    cv2.setNumThreads(2);bins=2**20;size=3024
    index={r['path']:r for r in read(folder/'index.json')}
    def mapped(row):
        a=load_map(index[row['path']])+np.float32(1)
        if a.min()<-1e-6 or a.max()>1.000001:raise RuntimeError('Unexpected FastFlow map range')
        return cv2.resize(a,(size,size),interpolation=cv2.INTER_LINEAR)
    hist=np.zeros(bins,np.int64)
    for row in source['calibration']:
        guard();a=mapped(row);q=np.minimum((a*(bins/2)).astype(np.int32),bins-1);hist+=np.bincount(q.ravel(),minlength=bins)
    threshold=(int(np.searchsorted(np.cumsum(hist),.99*hist.sum(),side='left'))+1)*2/bins
    del hist,a,q
    conditions=['same','background','illumination','view'];acc={c:RegionMetrics() for c in conditions}
    normals={c:[0,0] for c in conditions};regions=[];views=[]
    for row in source['evaluation']:
        guard();a=mapped(row);m=mask(row,size).astype(bool);c=row['condition'];acc[c].add(a,m)
        hits=a>threshold;ids,n=label(m,np.ones((3,3)));counts=np.bincount(ids.ravel())
        if not row['label']:normals[c][0]+=int(hits.sum());normals[c][1]+=hits.size
        views.append({'path':row['path'],'condition':c,'defect':row['defect'],'label':row['label'],'score':index[row['path']]['score']})
        for j in range(1,n+1):
            area=int(counts[j]);coverage=float(hits[ids==j].mean())
            regions.append({'path':row['path'],'region':j,'condition':c,'defect':row['defect'],'area':area,'small':area<=.001*m.size,'coverage':coverage,'detected':coverage>=.1})
    metrics={}
    for c in conditions:
        z=acc[c].result();pos=acc[c].positive[::-1].astype(float);neg=acc[c].negative[::-1].astype(float)
        tp=np.cumsum(pos)/pos.sum();fp=np.cumsum(neg)/neg.sum();subset=[r for r in views if r['condition']==c]
        z.update(pixel_auroc=float(np.sum((tp+np.r_[0,tp[:-1]])*np.diff(np.r_[0,fp])/2)),
            image_auroc=float(roc_auc_score([r['label'] for r in subset],[r['score'] for r in subset])),normal_pixel_fpr=normals[c][0]/normals[c][1])
        z['per_defect_image_auroc']={d:float(roc_auc_score([r['label'] for r in subset if r['defect'] in ('good',d)],
            [r['score'] for r in subset if r['defect'] in ('good',d)])) for d in ('ablation','breakdown','fracture','groove')}
        metrics[c]=z
    result={'conditions':metrics,'macro':{k:float(np.mean([metrics[c][k] for c in conditions])) for k in ('image_auroc','pixel_auroc','pixel_ap','aupro_005','normal_pixel_fpr')},
        'pixel_threshold':threshold,'threshold_bin_width':2/bins,'native_grid':[size,size],'development_only':True}
    write(folder/'metrics.json',result);write(folder/'regions.json',regions)
    del acc,ids,hits,m,a,pos,neg,tp,fp
    gallery(source['evaluation'],index,folder)
    initial=folder.parent/'epoch-000/regions.json'
    if folder.name=='epoch-500' and initial.exists():
        before={(r['path'],r['region']):r for r in read(initial)}
        lost=[r for r in regions if before[(r['path'],r['region'])]['detected'] and not r['detected']]
        gained=[r for r in regions if not before[(r['path'],r['region'])]['detected'] and r['detected']]
        write(folder.parent/'paired-regions.json',{'lost':lost,'gained':gained,'small_lost':sum(r['small'] for r in lost),'fracture_lost':sum(r['defect']=='fracture' for r in lost),
            'small_gained':sum(r['small'] for r in gained),'fracture_gained':sum(r['defect']=='fracture' for r in gained),'interpretation':'Compare changes with normal FPR; initialization is a diagnostic control, not a competitive method.'})
def evaluate(cfg,source,folder):
    guard()
    for row in source['masks']:verify_row(row)
    (aebad if cfg['dataset']=='aebad' else mvtec)(source,folder)

