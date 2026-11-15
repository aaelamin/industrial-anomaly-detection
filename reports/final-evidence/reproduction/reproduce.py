"""One configured command for fresh checks and all frozen paper experiment arms.

No model work occurs in plan or preflight. Reproduce executes fresh-output jobs;
it never attaches to the author's original results or checkpoint store.
"""
import argparse,ctypes,json,os,sys,time
from pathlib import Path
from suite_common import Suite,PACKAGE,read,write,sha,utc

def legacy(ctx,script,args,label,end,category='bottle',normal=False):
    command=[sys.executable,'-B','-u',str(PACKAGE/'legacy_entry.py'),'--config',str(ctx.config),
             '--script',script,'--category',category]
    if normal:command+=['--normal-only']
    command+=['--',*map(str,args)]
    with ctx.lock_gpu():return ctx.command(command,label,end)

def unchanged(folder):return {p.relative_to(folder).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()}

def bootstrap(ctx,end):
    destination=ctx.workspace/'bootstrap-result.json'
    if destination.exists():
        result=read(destination);assert result['suite_identity']==ctx.identity() and result['status']=='PASS_FRESH_OUTPUT_BOOTSTRAP'
        for name,h in result['evidence'].items():assert sha(ctx.workspace/name)==h,name
        return result
    clock=ctx.workspace/'bootstrap-clock.json'
    if clock.exists():
        old=read(clock);assert old['suite_identity']==ctx.identity();end=min(end,old['deadline_unix'])
        ctx.conditions(end)
    else:
        end=min(end,time.time()+3600)
        write(clock,{'utc':utc(),'suite_identity':ctx.identity(),'deadline_unix':end,'wall_cap_seconds':3600})
    checks=ctx.workspace/'bootstrap-checks';checks.mkdir(exist_ok=True)
    base=[sys.executable,'-B','-u',str(PACKAGE/'week3_normal_check.py'),'--config',str(ctx.config)]
    with ctx.lock_gpu():
        ctx.command(base+['--branch','control'],'week3-normal-control',end)
        folder=ctx.workspace/'week3-normal/interrupted'
        if not (folder/'injected-crash.json').exists():
            ctx.command(base+['--branch','interrupted','--crash'],'week3-normal-crash',end,91)
            ctx.command([sys.executable,'-B',str(PACKAGE/'inspect_crash.py'),'--folder',str(folder),'--kind','week3','--out',str(checks/'week3-crash.json')],'week3-inspect-crash',end)
        assert read(checks/'week3-crash.json')['status']=='PASS_ACTUAL_PRECOMMIT_STATE'
        crash=read(folder/'injected-crash.json');assert crash['exit']==91 and crash['checkpoint_epoch']==1 and crash['before_commit']
        ctx.command(base+['--branch','interrupted'],'week3-normal-resume',end)
        ctx.command([sys.executable,'-B',str(PACKAGE/'compare_states.py'),'--left',str(ctx.workspace/'week3-normal/control'),
                     '--right',str(folder),'--state','checkpoint.pt','--out',str(checks/'week3.json')],'week3-compare',end)
        before=unchanged(folder);ctx.command(base+['--branch','interrupted'],'week3-noop',end);assert unchanged(folder)==before
    store=ctx.project/'results/week4';scratch=ctx.project/'work/week4-cache'
    common=['--store',store,'--data',ctx.roots['mvtec'],'--scratch',scratch]
    legacy(ctx,'scripts/week4_colab.py',['--mode','recovery-save',*common],'week4-recovery-save',end,normal=True)
    legacy(ctx,'scripts/week4_colab.py',['--mode','recovery-check',*common],'week4-recovery-check',end,normal=True)
    w4=read(store/'recovery/passed.json');assert w4['continued_epoch_matches_exactly']
    # This original smoke command evaluates the already declared MVTec bottle
    # test split. Its metrics are implementation checks, not new paper results.
    legacy(ctx,'scripts/week5_study.py',['--smoke'],'week5-smoke',end)
    w5=read(ctx.project/'results/week5/study-smoke/summary.json')
    assert w5['complete'] and w5['recovery_exact'] and w5['source_sha256']==sha(ctx.project/'scripts/week5_study.py')
    added_config=ctx.workspace/'additional-host.json';rows=[]
    for job in ctx.data['diagnostic']['jobs']:
        jid=job['id'];base=[sys.executable,'-B','-u',str(PACKAGE/'additional/runner.py'),'normal-check',
              '--config',str(added_config),'--job',jid,'--wall-minutes','10']
        ctx.command(base+['--branch','control'],jid+'-control',end)
        folder=ctx.workspace/'additional-output/diagnostic'/jid/'interrupted'
        if not (folder/'injected-crash.json').exists():
            ctx.command(base+['--branch','interrupted','--inject-crash'],jid+'-crash',end,91)
            ctx.command([sys.executable,'-B',str(PACKAGE/'inspect_crash.py'),'--folder',str(folder),'--kind',job['kind'],'--out',str(checks/(jid+'-crash.json'))],jid+'-inspect-crash',end)
        assert read(checks/(jid+'-crash.json'))['status']=='PASS_ACTUAL_PRECOMMIT_STATE'
        crash=read(folder/'injected-crash.json')
        assert crash.get('before_memory_commit') if job['kind']=='patchcore' else crash.get('before_checkpoint_commit') and crash['epoch']==2
        ctx.command(base+['--branch','interrupted'],jid+'-resume',end)
        comparison=checks/(jid+'.json');state='memory.pt' if job['kind']=='patchcore' else 'resume.pt'
        ctx.command([sys.executable,'-B',str(PACKAGE/'compare_states.py'),'--left',str(folder.parent/'control'),
                     '--right',str(folder),'--state',state,'--out',str(comparison)],jid+'-compare',end)
        before=unchanged(folder);ctx.command(base+['--branch','interrupted'],jid+'-noop',end);assert unchanged(folder)==before
        row=read(comparison);row.update(job=jid,group=job['group'],recovery_equal=True,completed_noop=True,crash_exit=91);rows.append(row)
    q={'status':'PASS_NORMAL_ONLY_RELEASE_EQUIVALENCE','release_identity':ctx.manifest['additional_release_identity'],
       'rows':rows,'utc':utc(),'reserved_images_opened':0,'clean_machine_verified':False,'production_execution_verified':False}
    write(ctx.workspace/'additional-qualification.json',q)
    evidence=[store/'recovery/passed.json',ctx.project/'results/week5/study-smoke/summary.json',ctx.workspace/'additional-qualification.json']
    evidence+=list(checks.glob('*.json'))
    canon=lambda path:os.path.normcase(str(Path(path).resolve()))
    normal={canon(ctx.roots['mvtec']/r['path']) for r in ctx.data['datasets']['mvtec']['bottle']['train']}
    diagnostic={canon(ctx.roots['mvtec']/r['path']) for r in ctx.data['diagnostic']['normal_rows']}
    assert diagnostic<=normal
    bottle={canon(ctx.roots['mvtec']/r['path']) for role in ['train','calibration','evaluation','masks'] for r in ctx.data['datasets']['mvtec']['bottle'][role]}
    access_files=list((ctx.workspace/'legacy-access').glob('*.json'))
    access_files+=list((ctx.workspace/'week3-normal').glob('*/data-access*.json'))
    access_files+=list((ctx.workspace/'additional-output/diagnostic').glob('*/*/data-access.json'))
    assert len(access_files)>=14,'Missing access receipts'
    for path in access_files:
        access=read(path);assert access['reserved_images_opened']==0
        is_smoke=access.get('script')=='scripts/week5_study.py'
        boundary=diagnostic if path.is_relative_to(ctx.workspace/'additional-output') else (bottle if is_smoke else normal)
        assert {canon(p) for p in access['opened']} <= boundary,str(path)
        evidence.append(path)
    write(checks/'access-summary.json',{'status':'PASS_DECLARED_INPUTS_ONLY','receipts':len(access_files),'reserved_images_opened':0,'smoke_uses_declared_bottle_test':True})
    evidence.append(checks/'access-summary.json')
    result={'status':'PASS_FRESH_OUTPUT_BOOTSTRAP','utc':utc(),'suite_identity':ctx.identity(),
            'week3_normal_state_and_recovery':True,'week4_fresh_process_recovery':True,'week5_original_smoke_and_recovery':True,
            'additional_normal_routes':4,'source_images':'Existing MVTec bottle training normals; original Week 5 smoke also uses the declared 83 bottle test images and masks.',
            'reserved_images_opened':0,'clean_install_verified':False,'full_production_rerun_verified':False,
            'evidence':{p.relative_to(ctx.workspace).as_posix():sha(p) for p in evidence}}
    write(destination,result);return result

def production(ctx,end):
    bootstrap(ctx,end)
    completed=[]
    cats=list(ctx.data['datasets']['mvtec'])
    for group,script in [('week3','week3.py'),('week4','scripts/week4_colab.py')]:
        for category in cats:
            for seed in [0,1,2]:
                label=f'{group}-{category}-s{seed}';receipt=ctx.workspace/'suite-completions'/(label+'.json')
                if receipt.exists():
                    old=read(receipt);assert old['suite_identity']==ctx.identity()
                    for name,h in old['files'].items():assert sha(ctx.workspace/name)==h
                    completed.append(label);continue
                if group=='week3':
                    args=['--category',category,'--seeds',str(seed),'--data',ctx.roots['mvtec'],'--output',ctx.project/'results/week3']
                    folder=ctx.project/'results/week3'/category/f'seed_{seed}'
                else:
                    args=['--mode','run','--store',ctx.project/'results/week4','--data',ctx.roots['mvtec'],
                          '--scratch',ctx.project/'work/week4-cache','--category',category,'--seed',str(seed)]
                    folder=ctx.project/'results/week4/runs'/category/f'seed_{seed}'
                legacy(ctx,script,args,label,end,category)
                assert (folder/'metrics.json').exists()
                write(receipt,{'utc':utc(),'suite_identity':ctx.identity(),'files':{p.relative_to(ctx.workspace).as_posix():sha(p) for p in folder.rglob('*') if p.is_file()}})
                completed.append(label)
    legacy(ctx,'scripts/week5_study.py',[],'week5-full-study',end,category='all')
    assert read(ctx.project/'results/week5/study/summary.json')['complete']
    for job in ctx.data['jobs']:
        ctx.command([sys.executable,'-B','-u',str(PACKAGE/'additional/runner.py'),'run',
             '--config',str(ctx.workspace/'additional-host.json'),'--job',job['id'],'--wall-minutes','60',
             '--qualification',str(ctx.workspace/'additional-qualification.json')],job['id'],end)
    result={'status':'ALL_DECLARED_ARMS_COMPLETE','utc':utc(),'suite_identity':ctx.identity(),'planned_jobs':ctx.manifest['planned_jobs']}
    write(ctx.workspace/'suite-complete.json',result);return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['plan','preflight','bootstrap','reproduce'])
    p.add_argument('--config',required=True);p.add_argument('--wall-hours',type=float,default=1)
    a=p.parse_args();ctx=Suite(a.config)
    for path in reversed(ctx.extra):sys.path.insert(0,str(path))
    identity=ctx.check_sources(packages=a.mode!='plan')
    if a.mode=='plan':
        print(json.dumps({'status':'PASS_SUITE_PLAN','suite_identity':identity,'jobs':ctx.manifest['planned_jobs'],
                          'total':sum(ctx.manifest['planned_jobs'].values()),'model_execution':False,'images_opened':0}));return
    ctx.verify_weights()
    if a.mode=='preflight':print(json.dumps({'status':'PASS_SUITE_SOURCES_PACKAGES_WEIGHTS','suite_identity':identity,'images_opened':0}));return
    if not 0<a.wall_hours<=72:raise ValueError('Choose an explicit positive wall cap no greater than 72 hours')
    if a.mode=='bootstrap' and a.wall_hours>1:raise ValueError('The qualification cap is one hour')
    end=time.time()+a.wall_hours*3600
    if os.environ.get('STUDY_DEADLINE_UNIX'):end=min(end,float(os.environ['STUDY_DEADLINE_UNIX']))
    ctx.conditions(end)
    if os.name!='nt':raise RuntimeError('Only the Windows/CUDA route is supported')
    ctx.materialize()
    import msvcrt
    lock=(ctx.workspace/'suite-controller.lock').open('a+b');lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    from ctypes import wintypes
    class Memory(ctypes.Structure):
        _fields_=[('length',wintypes.DWORD),('load',wintypes.DWORD)]+[(n,ctypes.c_ulonglong) for n in ['total','available','total_page','available_page','total_virtual','available_virtual','extended']]
    m=Memory();m.length=ctypes.sizeof(m);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    import shutil,subprocess
    assert m.available>=5*2**30 and shutil.disk_usage(ctx.workspace).free>=8*2**30,'Need 5 GiB RAM and 8 GiB disk'
    assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())>=4096,'Need 4 GiB free GPU memory'
    previous=ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);assert previous
    write(ctx.workspace/'suite-current.json',{'status':'running','mode':a.mode,'deadline_unix':end,'suite_identity':identity,'pid':os.getpid()})
    try:
        result=bootstrap(ctx,end) if a.mode=='bootstrap' else production(ctx,end)
        print(json.dumps(result))
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(previous);lock.close()
        write(ctx.workspace/'suite-current.json',{'status':'closed','mode':a.mode,'deadline_unix':end,'suite_identity':identity,'utc':utc()})
if __name__=='__main__':main()
