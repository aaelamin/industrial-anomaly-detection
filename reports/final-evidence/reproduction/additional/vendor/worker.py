"""Resumable flow/coreset worker; endpoint and data come only from frozen manifests."""
from common import *
import argparse,random,copy,gc
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from feature_fastflow import FeatureFastFlow
from anomalib.models.image.fastflow.loss import FastflowLoss

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
def state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('feature_extractor.')}
def restore_model(model,saved):
    if set(saved)!=set(state(model)):raise RuntimeError('Checkpoint tensor keys differ')
    result=model.load_state_dict(saved,strict=False)
    if result.unexpected_keys or any(not k.startswith('feature_extractor.') for k in result.missing_keys):raise RuntimeError('Invalid state')
def tensor_digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def encoder_digest(model):return digest({k:tensor_digest(v) for k,v in model.feature_extractor.state_dict().items()})
def save_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
    with tmp.open('wb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    tmp.replace(path)
def rng_state(generator):
    n=np.random.get_state()
    return {'generator':generator.get_state(),'python':random.getstate(),'numpy':[n[0],n[1].tolist(),n[2],n[3],n[4]],'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}
def restore_rng(s,generator):
    generator.set_state(s['generator']);random.setstate(s['python']);n=s['numpy'];np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])
def load_checkpoint(path,identity,encoder_sha256):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    if saved['identity']!=identity or saved['encoder_sha256']!=encoder_sha256:raise RuntimeError('Resume identity mismatch')
    return saved
def load_weights(extractor,cfg):
    if sha(cfg['weights'])!=cfg['weights_sha256']:raise RuntimeError('Weight hash mismatch')
    # Only this fixed official PyTorch artifact needs trusted legacy deserialization.
    legacy=cfg['weights_sha256']=='5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8'
    w=torch.load(cfg['weights'],map_location='cpu',weights_only=not legacy);expected=extractor.state_dict()
    missing=set(expected)-set(w)
    if missing and not (legacy and all(k.endswith('num_batches_tracked') for k in missing)):raise RuntimeError(f'Missing weight keys {missing}')
    extractor.load_state_dict({k:w[k] if k in w else torch.zeros_like(v) for k,v in expected.items()},strict=True)
    extractor.requires_grad_(False).eval()
def build_flow(cfg):
    model=FeatureFastFlow(cfg['backbone'],conv3x3_only=cfg['conv3x3_only'])
    load_weights(model.feature_extractor,cfg);return model.cuda()
TRANSFORM=transforms.Compose([transforms.Resize((256,256),interpolation=transforms.InterpolationMode.BILINEAR),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
def images(rows):
    values=[]
    for row in rows:
        verify_row(row)
        with Image.open(row['path']) as im:values.append(TRANSFORM(im.convert('RGB')))
    return torch.stack(values).cuda()
def features(model,rows):
    arrays=None
    with torch.inference_mode():
        for offset in range(0,len(rows),4):
            guard();values=model.feature_extractor(images(rows[offset:offset+4]))
            if arrays is None:
                required=sum(len(rows)*v[0].numel()*4 for v in values)
                if resources()['ram_available']<required+int(1.25*2**30):raise RuntimeError('Insufficient RAM for category features')
                arrays=[torch.empty((len(rows),*v.shape[1:]),dtype=torch.float32) for v in values]
            for dst,v in zip(arrays,values):dst[offset:offset+len(v)].copy_(v.cpu())
    return arrays
def step(model,opt,cache,indices,micro):
    opt.zero_grad(set_to_none=True);total=0.;criterion=FastflowLoss()
    for idx in indices.split(micro):
        guard();loss=criterion(*model.from_features([v[idx].cuda() for v in cache]))
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        weighted=loss*(len(idx)/32);weighted.backward();total+=float(weighted.detach())
    if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise RuntimeError('Nonfinite gradient')
    opt.step();return total
def probe(model,rows):
    model.eval()
    with torch.inference_mode():return model(images(rows[:4])).anomaly_map.cpu()
def predict(model,rows,folder,identity):
    folder.mkdir(parents=True,exist_ok=True);model.eval();index=[]
    with torch.inference_mode():
        # Per-image commits permit deterministic recovery from interrupted evaluation.
        for row in rows:
            guard();name=key(row);item=folder/f'{name}.json';mapfile=folder/f'{name}.npy'
            if item.exists():
                saved=read(item)
                if saved['identity']!=identity or saved['map_sha256']!=sha(mapfile):raise RuntimeError('Corrupt/mismatched prediction')
            else:
                result=model(images([row]));a=result.anomaly_map[0,0].cpu().numpy()
                if not np.isfinite(a).all():raise RuntimeError('Nonfinite map')
                tmp=mapfile.with_name(mapfile.name+'.partial')
                with tmp.open('wb') as f:np.save(f,a,allow_pickle=False)
                tmp.replace(mapfile)
                saved={'identity':identity,'path':row['path'],'map_file':str(mapfile),'map_sha256':sha(mapfile),'score':float(result.pred_score.reshape(-1)[0].cpu())}
                write(item,saved)
            index.append(saved)
    write(folder/'index.json',index);return index
def train_flow(cfg,rows,out,identity,epochs,crash_epoch=None,diagnostic=False,eval_rows=None):
    seed_all(cfg['seed']);model=build_flow(cfg);encoder_before=encoder_digest(model)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=1e-5)
    gen=torch.Generator().manual_seed(cfg['seed']);resume=out/'resume.pt';history=[];done=0
    cache=features(model,rows)
    if resume.exists():
        saved=load_checkpoint(resume,identity,encoder_before)
        restore_model(model,saved['model']);opt.load_state_dict(copy.deepcopy(saved['optimizer']));restore_rng(saved['rng'],gen)
        done=saved['epoch'];history=saved['history'];del saved
    else:
        if cfg['dataset']=='aebad' and not diagnostic:predict(model,eval_rows,out/'epoch-000',identity)
        save_torch(resume,{'identity':identity,'model':state(model),'optimizer':opt.state_dict(),'rng':rng_state(gen),'epoch':0,'history':[],'encoder_sha256':encoder_before})
    if done>epochs:raise RuntimeError('Checkpoint beyond fixed endpoint')
    for epoch in range(done+1,epochs+1):
        guard();model.train();order=torch.randperm(len(rows),generator=gen);losses=[];started=time.monotonic()
        for offset in range(0,len(order)//32*32,32):losses.append(step(model,opt,cache,order[offset:offset+32],cfg['microbatch']))
        if not losses:raise RuntimeError('No complete batch')
        history.append({'epoch':epoch,'losses':losses,'shuffle_sha256':tensor_digest(order)})
        if crash_epoch==epoch:
            write(out/'injected-crash.json',{'epoch':epoch,'before_checkpoint_commit':True});os._exit(91)
        save_torch(resume,{'identity':identity,'model':state(model),'optimizer':opt.state_dict(),'rng':rng_state(gen),'epoch':epoch,'history':history,'encoder_sha256':encoder_before})
        write(out/'progress.json',{'epoch':epoch,'target_epochs':epochs,'seconds':time.monotonic()-started,'resources':resources()})
    if encoder_digest(model)!=encoder_before:raise RuntimeError('Encoder changed')
    save_torch(out/'normal-probe.pt',probe(model,rows))
    write(out/'training.json',history)
    del cache,opt;gc.collect();torch.cuda.empty_cache()
    return model
class BackboneAdapter(torch.nn.Module):
    def __init__(self,extractor):super().__init__();self.extractor=extractor
    def forward(self,x):
        f=self.extractor(x);return {'layer2':f[1],'layer3':f[2]}
def fit_patch(cfg,rows,out,identity,crash=False):
    import timm
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel
    from anomalib.models.components import KCenterGreedy
    seed_all(cfg['seed']);extractor=timm.create_model('wide_resnet50_2',pretrained=False,features_only=True,out_indices=[1,2,3]);load_weights(extractor,cfg)
    model=PatchcoreModel(['layer2','layer3'],backbone='wide_resnet50_2',pre_trained=False,num_neighbors=1)
    model.feature_extractor=BackboneAdapter(extractor);model=model.cuda().eval().requires_grad_(False)
    path=out/'memory.pt'
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['identity']!=identity:raise RuntimeError('Memory identity differs')
        model.memory_bank=saved['bank'].cuda()
    else:
        bank=torch.empty((len(rows)*32*32,1536),device='cuda');offset=0
        with torch.inference_mode():
            for i in range(0,len(rows),4):
                guard();f={k:model.feature_pooler(v) for k,v in model.feature_extractor(images(rows[i:i+4])).items()}
                emb=model.reshape_embedding(model.generate_embedding(f));bank[offset:offset+len(emb)].copy_(emb);offset+=len(emb)
        assert offset==len(bank);seed_all(cfg['seed']);sampler=KCenterGreedy(bank,sampling_ratio=.01);idx=sampler.select_coreset_idxs()
        if len(set(idx))!=len(idx):raise RuntimeError('Repeated coreset index')
        if crash:write(out/'injected-crash.json',{'before_memory_commit':True});os._exit(91)
        model.memory_bank=bank[idx].clone();save_torch(path,{'identity':identity,'bank':model.memory_bank.cpu(),'indices':idx})
        del bank,sampler,f,emb;gc.collect();torch.cuda.empty_cache()
    save_torch(out/'normal-probe.pt',probe(model,rows));return model
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',required=True);parser.add_argument('--diagnostic',action='store_true');parser.add_argument('--branch',choices=['control','interrupted']);parser.add_argument('--crash',action='store_true');args=parser.parse_args()
    frozen=validate();study=read(HERE/'study.json');data=read(HERE/'data-manifest.json')
    if args.diagnostic:
        diag=read(HERE/'diagnostic.json');cfg=next(x for x in diag['jobs'] if x['id']==args.job)
        rows=diag['normal_rows'][:8] if cfg['kind']=='patchcore' else diag['normal_rows'];eval_rows=[];masks=[]
        out=HERE/diag['output_directory']/cfg['id']/args.branch;epochs=3;identity=digest({'freeze':frozen,'diagnostic':diag,'job':cfg})
    else:
        approval=read(HERE/'launch-approval.json')
        if approval['freeze_sha256']!=sha(HERE/'freeze.json') or not approval.get('approved'):raise PermissionError('Full study not authorized')
        cfg=next(x for x in study['jobs'] if x['id']==args.job)
        if cfg['group'] not in approval['groups']:raise PermissionError('Group not authorized')
        source=data['aebad'] if cfg['dataset']=='aebad' else data['mvtec'][cfg['category']]
        rows=source['train'];eval_rows=source['calibration']+source['evaluation'];masks=source['masks']
        out=HERE/'runs'/cfg['id'];epochs=500;identity=digest({'freeze':frozen,'job':cfg})
    out.mkdir(parents=True,exist_ok=True);handle=lock_gpu();guard()
    allowed=rows if args.diagnostic else rows+eval_rows+masks
    opened=access_guard(data['roots'],allowed)
    complete=out/'complete.json'
    if complete.exists():
        c=read(complete)
        if c['identity']!=identity:raise RuntimeError('Completed identity differs')
        for path,h in c['artifacts'].items():
            if sha(path)!=h:raise RuntimeError('Completed artifact changed')
        print('Already complete; verified no-op',flush=True);return
    torch.cuda.reset_peak_memory_stats()
    try:
        if cfg['kind']=='flow':model=train_flow(cfg,rows,out,identity,epochs,2 if args.crash else None,args.diagnostic,eval_rows)
        else:model=fit_patch(cfg,rows,out,identity,args.crash)
        if not args.diagnostic:
            endpoint=out/'epoch-500' if cfg['kind']=='flow' else out/'evaluation';predict(model,eval_rows,endpoint,identity)
            del model;gc.collect();torch.cuda.empty_cache()
            from evaluation import evaluate
            for folder in ([out/'epoch-000',endpoint] if cfg['dataset']=='aebad' else [endpoint]):evaluate(cfg,source,folder)
        write(out/'data-access.json',{'opened':sorted(opened),'allowlist_only':True,'reserved_images_opened':0})
        artifacts={str(p):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ('complete.json','progress.json','injected-crash.json') and not p.name.endswith('.partial')}
        write(complete,{'identity':identity,'diagnostic':args.diagnostic,'artifacts':artifacts,'resources':resources(),'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20})
        print(json.dumps({'job':cfg['id'],'complete':True,'diagnostic':args.diagnostic}),flush=True)
    except BaseException as e:
        write(out/'last-failure.json',{'utc':utc(),'error':repr(e)});write(out/'data-access.json',{'opened':sorted(opened),'allowlist_only':True,'reserved_images_opened':0});raise
    finally:handle.close()
if __name__=='__main__':main()
