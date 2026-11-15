"""Two-epoch normal-only check using the original Week 3 training functions."""
import argparse, os, random, sys
from pathlib import Path
from suite_common import Suite, PACKAGE, read, write, sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--branch',choices=['control','interrupted'],required=True);p.add_argument('--crash',action='store_true');a=p.parse_args()
    ctx=Suite(a.config);ctx.check_sources();sys.path.insert(0,str(ctx.project))
    import week3 as w
    import torch,numpy as np
    cfg=read(ctx.project/'configs/week3.json');out=ctx.workspace/'week3-normal'/a.branch;out.mkdir(parents=True,exist_ok=True)
    identity=ctx.identity();complete=out/'complete.json'
    if complete.exists():
        r=read(complete);assert r['suite_identity']==identity
        for name,h in r['files'].items():assert sha(out/name)==h
        print('Verified completed normal-only no-op');return
    rows=ctx.data['datasets']['mvtec']['bottle']['train'];allowed={os.path.normcase(str((ctx.roots['mvtec']/r['path']).resolve())):r['sha256'] for r in rows}
    roots=[os.path.normcase(str(v)) for v in ctx.roots.values()];opened=set()
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=os.path.normcase(str(Path(os.fsdecode(args[0])).resolve()))
        if any(path.startswith(r+os.sep) for r in roots):
            if path not in allowed:raise PermissionError('Non-normal or undeclared image: '+path)
            flags=args[2] if len(args)>2 else 0
            if isinstance(flags,int) and flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):raise PermissionError('Dataset write')
            opened.add(path)
    sys.addaudithook(audit)
    for path,h in allowed.items():assert sha(path)==h
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    model=w.build_model(cfg,'cuda');train=w.MVTecDataset(ctx.roots['mvtec'],'bottle','train',256)
    assert {os.path.normcase(str(p.resolve())) for p in train.paths}==set(allowed)
    opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    gen=torch.Generator().manual_seed(0);features=w.cache_features(model,train,cfg['micro_batch_size'],'cuda')
    checkpoint=out/'checkpoint.pt';history=[];start=0
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True);assert saved['suite_identity']==identity
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);gen.set_state(saved['generator'])
        start=saved['epoch'];history=saved['history'];del saved
    for epoch in range(start,2):
        loss=w.train_epoch(model,features,opt,gen,cfg,'cuda');history.append(loss)
        if a.crash and epoch==1:
            write(out/'injected-crash.json',{'exit':91,'after_epoch':2,'checkpoint_epoch':1,'before_commit':True})
            write(out/'data-access-before-crash.json',{'opened':sorted(opened),'normal_only':True,'reserved_images_opened':0})
            os._exit(91)
        tmp=out/'checkpoint.partial'
        torch.save({'suite_identity':identity,'model':model.state_dict(),'optimizer':opt.state_dict(),'generator':gen.get_state(),'epoch':epoch+1,'history':history},tmp);tmp.replace(checkpoint)
    model.eval()
    with torch.inference_mode():probe=model(torch.stack([train[i][0] for i in range(4)]).cuda()).anomaly_map.cpu()
    assert torch.isfinite(probe).all();torch.save(probe,out/'normal-probe.pt')
    write(out/'data-access.json',{'opened':sorted(opened),'normal_only':True,'reserved_images_opened':0})
    write(complete,{'suite_identity':identity,'epochs':2,'images':len(train),'files':{n:sha(out/n) for n in ['checkpoint.pt','normal-probe.pt','data-access.json']}})
    print('Completed original Week 3 normal-only check')
if __name__=='__main__':main()
