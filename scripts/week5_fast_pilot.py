"""Validate microbatch 32 using the same checks as the original pilot.

Frozen companion to week5_pilot.py, with separate outputs and image batches of four.
"""
import argparse
import ctypes
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.week5_model import ProjectedFlow, Projection, fit_projection, train_epoch, seed_all, torch, np
from scripts.week4_colab import (write_json, digest, state_hash, rng_state, restore_rng,
                                 save_checkpoint, load_checkpoint)
from scripts.week4_preflight import cache_on_disk, DiskFeatures
from week3 import build_model, MVTecDataset, DataLoader, file_hash
from run import CATEGORIES

STORE=ROOT/'results/week5/fast-pilot'
WORK=ROOT/'work/week5-fast-pilot'
REPORT=ROOT/'reports/week5/fast'


def read(path):
    return json.loads(path.read_text())


def identity(config):
    names=['scripts/week5_pilot.py','configs/week5_fast.json','configs/week4.json','scripts/week5_model.py',
           'scripts/week5_fast_pilot.py','scripts/week4_colab.py','scripts/week4_preflight.py','week3.py','run.py']
    return {'config':config,'sources':{n:file_hash(ROOT/n) for n in names},
            'packages':{n:importlib.metadata.version(n) for n in ['torch','torchvision','anomalib','timm','numpy','scikit-learn']}}


def optimizer(model,config):
    return torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
        lr=config['learning_rate'],weight_decay=config['weight_decay'])


def fingerprint(model,opt,generator):
    return state_hash({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(generator)})


def payload(model,opt,generator,history,protocol,projection_sha):
    return {'epoch':len(history),'model':model.state_dict(),'optimizer':opt.state_dict(),
        'rng':rng_state(generator),'history':history,'protocol':protocol,'projection_sha256':projection_sha}


def timed_epoch(model,features,opt,generator,config,epoch):
    torch.cuda.synchronize()
    tick=time.perf_counter()
    row=train_epoch(model,features,opt,generator,config)
    torch.cuda.synchronize()
    return {'epoch':epoch,**row,'seconds':time.perf_counter()-tick}


def verify_resume(args,config):
    seed_all(0)
    saved=load_checkpoint(args.checkpoint)
    expected=read(args.expected)
    if saved['protocol'] != expected['protocol']:
        raise ValueError('Recovery protocol mismatch.')
    model=ProjectedFlow(config).cuda()
    model.load_state_dict(saved['model'],strict=True)
    opt=optimizer(model,config)
    opt.load_state_dict(saved['optimizer'])
    generator=torch.Generator()
    restore_rng(saved['rng'],generator)
    features=DiskFeatures(args.features)
    actual=train_epoch(model,features,opt,generator,config)
    assert fingerprint(model,opt,generator)==expected['state_hash'], 'Resumed state differs'
    assert actual==expected['training'], 'Resumed training measurements differ'
    features.values._mmap.close()
    write_json(args.expected.with_name('recovery_passed.json'),{'fresh_process_resume_exact':True,
        'protocol':saved['protocol'],'epoch':saved['epoch']+1})
    print('Fresh-process continuation matched model, optimizer, random states and loss.',flush=True)


def prepare_category(backbone,category,config):
    tick=time.perf_counter()
    train=MVTecDataset(ROOT/'data/mvtec_ad',category,'train',256)
    scratch=WORK/f'{category}-{uuid.uuid4().hex[:8]}'
    scratch.mkdir(parents=True)
    output=STORE/category
    output.mkdir(parents=True,exist_ok=True)
    print(f'{category}: extracting frozen training features.',flush=True)
    wrapper=type('Extractor',(),{'feature_extractor':backbone})()
    features=cache_on_disk(wrapper,train,scratch,config['feature_batch_size'])
    extraction_seconds=time.perf_counter()-tick
    tick=time.perf_counter()
    training_hashes={p.relative_to(train.root).as_posix():file_hash(p) for p in train.paths}
    projection_path=output/'projection.pt'
    if projection_path.exists():
        receipt=read(output/'projection.json')
        assert receipt['sha256']==file_hash(projection_path)
        assert receipt['training_files']==training_hashes
        saved=torch.load(projection_path,map_location='cpu',weights_only=True)
    else:
        print(f'{category}: fitting training-only PCA projections.',flush=True)
        saved=fit_projection(features,train,config)
        assert all(torch.isfinite(level[key]).all() for level in saved['levels'] for key in ('mean','components','offset','scale'))
        torch.save(saved,projection_path)
        write_json(output/'projection.json',{'sha256':file_hash(projection_path),'fit_split':'train/good',
            'training_files':training_hashes,'samples_per_level':[len(p['sample_indices']) for p in saved['levels']],
            'floor_count_per_level':[p['floor_count'] for p in saved['levels']]})
    projection_seconds=time.perf_counter()-tick
    projection=Projection(saved,config['grid']).cuda()
    tick=time.perf_counter()
    arrays={arm:np.lib.format.open_memmap(scratch/f'{arm}.npy',mode='w+',dtype=np.float32,
        shape=(len(train),config['channels'],config['grid'],config['grid'])) for arm in config['arms']}
    with torch.inference_mode():
        for start in range(0,len(train),config['feature_batch_size']):
            ids=torch.arange(start,min(start+config['feature_batch_size'],len(train)))
            native=[f[ids].cuda() for f in features]
            for arm in config['arms']:
                values=projection(native,arm)
                assert torch.isfinite(values).all()
                arrays[arm][start:start+len(ids)]=values.cpu().numpy()
    for array in arrays.values():
        array.flush()
        array._mmap.close()
    projected_seconds=time.perf_counter()-tick
    # Check disk features against a fresh extraction from the original images.
    ids=torch.arange(config['feature_batch_size'])
    images=torch.stack([train[int(i)][0] for i in ids]).cuda()
    with torch.inference_mode():
        native=backbone(images)
        for arm in config['arms']:
            cached=DiskFeatures(scratch/f'{arm}.npy')
            torch.testing.assert_close(projection(native,arm).cpu(),cached[ids],rtol=0,atol=0)
            cached.values._mmap.close()
    raw_bytes=sum((scratch/f'level_{i}.npy').stat().st_size for i in range(3))
    for i,feature in enumerate(features):
        feature.values._mmap.close()
        (scratch/f'level_{i}.npy').unlink()
    info={'category':category,'train_images':len(train),'extraction_seconds':extraction_seconds,
        'projection_fit_seconds':projection_seconds,'projected_cache_seconds':projected_seconds,
        'native_feature_bytes':raw_bytes,'projected_cache_bytes':sum((scratch/f'{arm}.npy').stat().st_size for arm in config['arms']),
        'projection_bytes':projection_path.stat().st_size,'projected_features_match_fresh_extraction':True}
    write_json(output/'preparation.json',info)
    return scratch,projection,info,file_hash(projection_path)


def run_pilot(config):
    if config['pilot_epochs']!=10 or config['pilot_categories']!=['bottle','screw','carpet']:
        raise ValueError('This command is limited to the agreed ten-epoch pilot.')
    if shutil.disk_usage(ROOT).free < 8*1024**3:
        raise RuntimeError('Need at least 8 GiB free for one category and pilot checkpoints.')
    inventory=read(ROOT/'docs/DATA_CHECK.json')['categories']
    for row in inventory:
        base=ROOT/'data/mvtec_ad'/row['category']
        assert len(list((base/'train/good').glob('*.png')))==row['train']
        assert len(list((base/'test').glob('*/*.png')))==row['test']
    assert {row['category'] for row in inventory}==set(CATEGORIES)
    WORK.mkdir(parents=True,exist_ok=True)
    STORE.mkdir(parents=True,exist_ok=True)
    REPORT.mkdir(parents=True,exist_ok=True)
    # A second invocation must not write to the same pilot concurrently.
    lock=(WORK/'controller.lock').open('a+b')
    if os.name=='nt':
        import msvcrt
        lock.seek(0)
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    metadata=identity(config)
    protocol=digest(metadata)
    manifest=STORE/'study.json'
    if manifest.exists():
        assert read(manifest)==metadata,'Pilot code or settings changed. Use a separate output for a different pilot.'
    else:
        write_json(manifest,metadata)
    started=time.perf_counter()
    seed_all(0)
    backbone=build_model(read(ROOT/config['backbone_config']),'cuda').feature_extractor
    backbone.eval()
    backbone.requires_grad_(False)
    backbone_hash=state_hash(backbone.state_dict())
    baseline_memory=torch.cuda.memory_allocated()
    checks=[]
    preparation=[]
    try:
        for category in config['pilot_categories']:
            pending=[(s,a) for s in config['seeds'] for a in config['arms']
                if not (STORE/category/f'seed_{s}'/a/'complete.json').exists()]
            if not pending:
                preparation.append(read(STORE/category/'preparation.json'))
                for seed in config['seeds']:
                    for arm in config['arms']:
                        out=STORE/category/f'seed_{seed}'/arm
                        receipt=read(out/'complete.json')
                        assert receipt['protocol']==protocol
                        assert all(file_hash(out/n)==v for n,v in receipt['files'].items())
                        checks.append(read(out/'benchmark.json'))
                continue
            if shutil.disk_usage(ROOT).free < 8*1024**3:
                raise RuntimeError('Need 8 GiB free before preparing the next category.')
            scratch,projection,prep,projection_sha=prepare_category(backbone,category,config)
            preparation.append(prep)
            test=MVTecDataset(ROOT/'data/mvtec_ad',category,'test',256)
            initial_hashes={}
            for seed in config['seeds']:
                for arm in config['arms']:
                    out=STORE/category/f'seed_{seed}'/arm
                    out.mkdir(parents=True,exist_ok=True)
                    if (out/'complete.json').exists():
                        receipt=read(out/'complete.json')
                        assert receipt['protocol']==protocol
                        assert all(file_hash(out/n)==v for n,v in receipt['files'].items())
                        checks.append(read(out/'benchmark.json'))
                        continue
                    seed_all(seed)
                    model=ProjectedFlow(config).cuda()
                    initial=state_hash(model.state_dict())
                    if seed in initial_hashes:
                        assert initial==initial_hashes[seed]
                    initial_hashes[seed]=initial
                    opt=optimizer(model,config)
                    generator=torch.Generator().manual_seed(seed)
                    features=DiskFeatures(scratch/f'{arm}.npy')
                    checkpoint=out/'checkpoints'
                    saved=load_checkpoint(checkpoint)
                    history=[]
                    if saved:
                        assert saved['protocol']==protocol and saved['projection_sha256']==projection_sha
                        model.load_state_dict(saved['model'],strict=True)
                        opt.load_state_dict(saved['optimizer'])
                        restore_rng(saved['rng'],generator)
                        history=saved['history']
                        del saved
                    torch.cuda.reset_peak_memory_stats()
                    save_times=[]
                    while len(history)<config['pilot_epochs']:
                        epoch=len(history)+1
                        write_json(WORK/'status.json',{'state':'running','pid':os.getpid(),
                            'category':category,'seed':seed,'arm':arm,'epoch':epoch,'epochs':10,'completed_runs':len(checks)})
                        row=timed_epoch(model,features,opt,generator,config,epoch)
                        history.append(row)
                        # Test recovery in a new Python process before proceeding with the pilot.
                        if category=='bottle' and seed==0 and arm=='middle' and epoch==2:
                            expected=out/'recovery_expected.json'
                            write_json(expected,{'protocol':protocol,'state_hash':fingerprint(model,opt,generator),
                                'training':{k:v for k,v in row.items() if k not in ('seconds','epoch')}})
                            subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--verify-resume',
                                '--checkpoint',str(checkpoint),'--features',str(scratch/f'{arm}.npy'),
                                '--expected',str(expected)],cwd=ROOT,check=True)
                        tick=time.perf_counter()
                        save_checkpoint(checkpoint,payload(model,opt,generator,history,protocol,projection_sha))
                        save_times.append(time.perf_counter()-tick)
                        print(f'{category} seed {seed} {arm}: {epoch}/10, {row["seconds"]:.2f}s, loss {row["loss"]:.1f}',flush=True)
                    # Reconstruction of the final weights must preserve predictions exactly.
                    model.eval()
                    probe=features[torch.arange(config['feature_batch_size'])].cuda()
                    with torch.inference_mode():
                        before=model.anomaly_map(probe).cpu()
                    torch.save({'model':model.state_dict(),'config':config,'projection_sha256':projection_sha,
                        'backbone_config':read(ROOT/config['backbone_config']),'protocol':protocol,'pilot_only':True},out/'model.pt')
                    restored=ProjectedFlow(config).cuda()
                    restored.load_state_dict(torch.load(out/'model.pt',map_location='cpu',weights_only=True)['model'],strict=True)
                    restored.eval()
                    with torch.inference_mode():
                        after=restored.anomaly_map(probe).cpu()
                    torch.testing.assert_close(before,after,rtol=0,atol=0)
                    del restored,probe,before,after
                    torch.cuda.synchronize()
                    tick=time.perf_counter()
                    evaluated=0
                    with torch.inference_mode():
                        for images,*_ in DataLoader(test,batch_size=config['feature_batch_size']):
                            maps=model.anomaly_map(projection(backbone(images.cuda()),arm))
                            assert maps.shape==(len(images),1,256,256) and torch.isfinite(maps).all()
                            evaluated+=len(images)
                    torch.cuda.synchronize()
                    evaluation_seconds=time.perf_counter()-tick
                    assert evaluated==len(test)
                    assert state_hash(backbone.state_dict())==backbone_hash
                    latest=max(checkpoint.glob('*.pt'),key=lambda p:p.stat().st_mtime)
                    result={'category':category,'seed':seed,'arm':arm,'pilot_only':True,'epochs':10,
                        'initial_state_hash':initial,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                        'train_images':len(features),'steps_per_epoch':len(features)//config['batch_size'],
                        'median_epoch_seconds_after_first':statistics.median(r['seconds'] for r in history[1:]),
                        'checkpoint_save_seconds_median':statistics.median(save_times) if save_times else None,
                        'checkpoint_bytes':latest.stat().st_size,'final_model_bytes':(out/'model.pt').stat().st_size,
                        'evaluation_seconds':evaluation_seconds,'test_images':evaluated,
                        'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                        'gpu_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
                        'initial_backbone_mib':baseline_memory/1024**2,'gpu':torch.cuda.get_device_name(0),
                        'checkpoint_prediction_reload_exact':True,'backbone_unchanged':True,'protocol':protocol}
                    write_json(out/'training.json',history)
                    write_json(out/'benchmark.json',result)
                    write_json(out/'complete.json',{'protocol':protocol,'files':{n:file_hash(out/n)
                        for n in ['model.pt','training.json','benchmark.json']}})
                    checks.append(result)
                    features.values._mmap.close()
                    del model,opt,features,maps
                    gc.collect()
                    torch.cuda.empty_cache()
            del projection
            # Only these two files are temporary, and their exact directory was created above.
            assert scratch.resolve().parent==WORK.resolve()
            for arm in config['arms']:
                (scratch/f'{arm}.npy').unlink()
            scratch.rmdir()
        assert len(checks)==18
        for category in config['pilot_categories']:
            for seed in config['seeds']:
                pair=[r for r in checks if r['category']==category and r['seed']==seed]
                assert len({r['initial_state_hash'] for r in pair})==1
                assert len({r['trainable_parameters'] for r in pair})==1
        counts={c:len(list((ROOT/'data/mvtec_ad'/c/'train/good').glob('*.png'))) for c in CATEGORIES}
        test_counts={c:len(list((ROOT/'data/mvtec_ad'/c/'test').glob('*/*.png'))) for c in CATEGORIES}
        per_step=[r['median_epoch_seconds_after_first']/r['steps_per_epoch'] for r in checks]
        steps=sum(n//config['batch_size'] for n in counts.values())*config['full_epochs_for_estimate_only']*3*2
        train_hours=[v*steps/3600 for v in (min(per_step),statistics.median(per_step),max(per_step))]
        prep_per_image=[(p['extraction_seconds']+p['projected_cache_seconds'])/p['train_images'] for p in preparation]
        prep_seconds=statistics.median(prep_per_image)*sum(counts.values())+statistics.median(p['projection_fit_seconds'] for p in preparation)*15
        saves=[r['checkpoint_save_seconds_median'] for r in checks if r['checkpoint_save_seconds_median'] is not None]
        checkpoint_seconds=statistics.median(saves)*20*90
        inference_seconds=statistics.median(r['evaluation_seconds']/r['test_images'] for r in checks)*sum(test_counts.values())*6
        summary={'pilot_complete':True,'completed_runs':18,'full_study_started':False,'protocol':protocol,
            'fresh_process_resume':read(STORE/'bottle/seed_0/middle/recovery_passed.json'),
            'runs':checks,'preparation':preparation,'training_counts':counts,'test_counts':test_counts,
            'full_90_run_500_epoch_training_hours_low_median_high':train_hours,
            'estimated_shared_preparation_hours':prep_seconds/3600,
            'estimated_checkpoint_hours_every25epochs':checkpoint_seconds/3600,
            'estimated_model_inference_hours':inference_seconds/3600,
            'estimated_largest_category_temporary_feature_gib':max(counts.values())*(256*64*64+512*32*32+1024*16*16+2*96*64*64)*4/1024**3,
            'estimated_full_model_and_recovery_gib':statistics.median(r['final_model_bytes'] for r in checks)*90*4/1024**3 + statistics.median(r['checkpoint_bytes'] for r in checks)*2/1024**3,
            'elapsed_this_invocation_seconds':time.perf_counter()-started,
            'estimate_limits':'Observed range is not a confidence interval. Assumes category caches shared across seeds and arms. Inference excludes metric sorting, preview/map export and report generation. Storage excludes dataset, existing studies and any dense test maps. Long-run thermal throttling, interruptions and retries are not measured.'}
        write_json(REPORT/'pilot.json',summary)
        write_json(WORK/'status.json',{'state':'complete','completed_runs':18,'full_study_started':False})
        print(json.dumps({k:v for k,v in summary.items() if k not in ['runs','preparation','training_counts','test_counts']},indent=2),flush=True)
    except BaseException as error:
        write_json(WORK/'status.json',{'state':'stopped','error':str(error),'pid':os.getpid()})
        raise
    finally:
        if os.name=='nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--verify-resume',action='store_true')
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--features',type=Path)
    parser.add_argument('--expected',type=Path)
    args=parser.parse_args()
    config=read(ROOT/'configs/week5_fast.json')
    if args.verify_resume:
        verify_resume(args,config)
    else:
        run_pilot(config)
