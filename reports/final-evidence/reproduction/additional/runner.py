"""Configured qualification and receipt-gated execution of the additional-study worker."""
import argparse,builtins,json,os,subprocess,sys,time
from pathlib import Path
from configuration import Configuration,read,sha,digest

PACKAGE=Path(__file__).resolve().parent

def stop_owned(proc):
    import psutil
    try:children=psutil.Process(proc.pid).children(recursive=True)
    except psutil.NoSuchProcess:children=[]
    for child in reversed(children):
        try:child.kill()
        except psutil.NoSuchProcess:pass
    psutil.wait_procs(children,timeout=5)
    if proc.poll() is None:proc.kill()
    return proc.wait(timeout=5)

def initialize(ctx):
    identity=ctx.prepare_store()
    import runtime_common as c
    c.configure(ctx)
    data={'roots':list(map(str,ctx.roots.values())), 'mvtec':{},'aebad':{}}
    for dataset,sources in [('mvtec',ctx.manifest['datasets']['mvtec'].items()),('aebad',[('aebad',ctx.manifest['datasets']['aebad'])])]:
        for category,source in sources:
            rows={role:[ctx.resolved_row(r) for r in source[role]] for role in ['train','calibration','evaluation','masks']}
            if dataset=='mvtec':data[dataset][category]=rows
            else:data[dataset]=rows
    diagnostic={'jobs':[ctx.resolved_job(j) for j in ctx.manifest['diagnostic']['jobs']],
        'normal_rows':[ctx.resolved_row(r) for r in ctx.manifest['diagnostic']['normal_rows']],
        'epochs':3,'output_directory':'diagnostic','full_study':False}
    values={'study.json':{'jobs':[ctx.resolved_job(j) for j in ctx.manifest['jobs']]},
            'data-manifest.json':data,'diagnostic.json':diagnostic,'freeze.json':c.validate()}
    for name,value in values.items():
        target=ctx.output/name
        if target.exists() and ctx.canonical(read(target))!=ctx.canonical(value):
            raise ValueError('Stored manifest differs: '+name)
        c.write(target,value)
    return c,identity

def worker(ctx,args):
    if os.name!='nt':raise RuntimeError('This candidate training path is Windows/CUDA only')
    c,identity=initialize(ctx)
    sys.modules['common']=c
    vendor=PACKAGE/'vendor';sys.path.insert(0,str(vendor))
    from resource_gate import ResourceGate,GIB
    from scratch import Scratch
    import io_recovery
    io_recovery.HERE=ctx.scratch;io_recovery.STUDY=ctx.output;io_recovery.install()
    def event(value):c.write(ctx.output/'resource-events'/f'{time.time_ns()}-{os.getpid()}.json',{'utc':c.utc(),**value})
    gate=ResourceGate(c.resources,c.conditions,event);c._GATE=gate
    normal=not args.production_worker
    jobs=ctx.manifest['diagnostic']['jobs'] if normal else ctx.manifest['jobs']
    cfg=next(j for j in jobs if j['id']==args.job)
    if normal:
        allowed=read(ctx.output/'diagnostic.json')['normal_rows']
        if cfg['kind']=='patchcore':allowed=allowed[:8]
    else:
        data=read(ctx.output/'data-manifest.json')
        source=data['aebad'] if cfg['dataset']=='aebad' else data['mvtec'][cfg['category']]
        allowed=[r for role in ['train','calibration','evaluation','masks'] for r in source[role]]
    opened=c.access_guard(list(map(str,ctx.roots.values())),allowed)
    preflight=ctx.verify_inputs(normal_only=normal,job_id=args.job)
    c.write(ctx.output/'input-preflights'/f'{time.time_ns()}-{os.getpid()}.json',
            {**preflight,'opened':sorted(opened),'allowlist_only':True,'job':args.job})
    target=vendor/'worker.py';sys.argv=[str(target),'--job',args.job]
    if normal:sys.argv+=['--diagnostic','--branch',args.branch]
    if args.inject_crash:sys.argv.append('--crash')
    original_import=builtins.__import__
    def importing(name,*pos,**kw):
        module=original_import(name,*pos,**kw)
        if name=='evaluation' and Path(module.__file__).resolve()==(vendor/'evaluation.py').resolve() and not getattr(module,'_release_gate',False):
            evaluate_original=module.evaluate
            def evaluate(config,source,folder):
                needed=(len(source['evaluation'])*256*256*40+GIB) if config['dataset']=='mvtec' else 1.5*GIB
                with gate.allocation(needed,needed+.25*GIB,stage='exact_metrics'):
                    gate.check();return evaluate_original(config,source,folder)
            module.evaluate=evaluate;module._release_gate=True
        return module
    namespace={'__file__':str(target),'__name__':'release_worker',
               '__builtins__':{**vars(builtins),'__import__':importing}}
    exec(compile(target.read_text(encoding='utf-8'),str(target),'exec'),namespace)
    original_features=namespace['features'];original_train=namespace['train_flow'];lease=None;scratch=None
    def features(model,rows):
        nonlocal lease,scratch
        scratch=Scratch(ctx.scratch,args.job);scratch.recover();lease=scratch.create()
        from cache_io import wrapper
        with gate.allocation(1.5*GIB,1.75*GIB,stage='feature_cache'):
            gate.check()
            return wrapper(original_features,namespace,'mapped',lease)(model,rows)
    def train(*pos,**kw):
        try:return original_train(*pos,**kw)
        finally:
            if lease is not None:scratch.clean(lease,current=True)
    namespace['features']=features;namespace['train_flow']=train
    namespace['main']()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['preflight','normal-check','run','_worker'])
    p.add_argument('--config',required=True,help='Private JSON containing explicit data/weight/output/scratch roots')
    p.add_argument('--job',help='Declared diagnostic job ID')
    p.add_argument('--branch',choices=['control','interrupted'],default='control')
    p.add_argument('--wall-minutes',type=float,default=10)
    p.add_argument('--inject-crash',action='store_true')
    p.add_argument('--qualification',help='Matching normal-only equivalence/recovery receipt, required by run')
    p.add_argument('--production-worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--metadata-only',action='store_true',help='No model imports, input hashes or output creation')
    a=p.parse_args();ctx=Configuration(PACKAGE,a.config)
    ctx.verify_sources(packages=not a.metadata_only)
    if a.command=='preflight':
        result={'status':'PASS_RELEASE_METADATA','release_identity':ctx.fingerprint(),'model_execution':False,
                'source_dataset_images_opened':0,'production_qualified':False}
        if not a.metadata_only:
            result.update(ctx.verify_inputs(normal_only=True,job_id=a.job));result['status']='PASS_NORMAL_INPUT_PREFLIGHT'
            result['source_dataset_images_opened']=result['input_hashes']
        print(json.dumps(result));return
    if a.metadata_only:raise ValueError('Metadata-only is a preflight option')
    production=a.command=='run' or (a.command=='_worker' and a.production_worker)
    jobs=ctx.manifest['jobs'] if production else ctx.manifest['diagnostic']['jobs']
    if a.job not in {j['id'] for j in jobs}:raise ValueError('A declared job is required')
    if production:
        if not a.qualification:raise ValueError('Production disabled: a verified route-specific qualification receipt is required')
        q=read(a.qualification);cfg=next(j for j in jobs if j['id']==a.job)
        if q.get('status')!='PASS_NORMAL_ONLY_RELEASE_EQUIVALENCE' or q.get('release_identity')!=ctx.fingerprint():
            raise ValueError('Production disabled: qualification identity/status differs')
        routes={r['group'] for r in q['rows'] if r.get('bitwise_state') and r.get('bitwise_probe') and r.get('recovery_equal') and r.get('completed_noop')}
        if cfg['group'] not in routes:raise ValueError('Production disabled: this route is unqualified')
    if a.command=='_worker':
        if not os.environ.get('STUDY_DEADLINE_UNIX'):raise ValueError('Internal worker requires bounded parent')
        worker(ctx,a);return
    if not 0<a.wall_minutes<=60:raise ValueError('Each public launch is capped at one hour; resume the same identity afterwards')
    import runtime_common as c
    c,identity=initialize(ctx)
    r=c.resources()
    if r['ram_available']<3*2**30 or r['disk_free']<8*2**30:raise RuntimeError('Launch needs 3 GiB RAM and 8 GiB disk')
    gpu_free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    if gpu_free<4096:raise RuntimeError('Launch needs 4 GiB free GPU memory')
    probe=c.lock_gpu();probe.close()
    end=time.time()+a.wall_minutes*60
    parent_deadline=os.environ.get('STUDY_DEADLINE_UNIX')
    if parent_deadline:end=min(end,float(parent_deadline))
    if time.time()>=end-15:raise RuntimeError('No authorized time remains')
    env=os.environ.copy();env['STUDY_DEADLINE_UNIX']=str(end-10)
    command=[sys.executable,'-B','-u',str(PACKAGE/'runner.py'),'_worker','--config',str(Path(a.config).resolve()),
             '--job',a.job,'--branch',a.branch]
    if a.inject_crash:command.append('--inject-crash')
    if production:
        if a.inject_crash:raise ValueError('Crash injection is restricted to diagnostics')
        command+=['--production-worker','--qualification',str(Path(a.qualification).resolve())]
        c.write(ctx.output/'launch-approval.json',{'approved':True,'freeze_sha256':sha(ctx.output/'freeze.json'),
                'groups':[cfg['group']],'expires_unix':end,'release_identity':identity})
    logs=ctx.output/'logs';logs.mkdir(exist_ok=True);stamp=str(time.time_ns())
    proc=None;failure=None;code=None
    try:
        with (logs/(stamp+'.out')).open('x') as out,(logs/(stamp+'.err')).open('x') as err:
            proc=subprocess.Popen(command,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            try:code=proc.wait(timeout=max(1,end-time.time()))
            except subprocess.TimeoutExpired:
                failure='wall_cap';code=stop_owned(proc)
    except BaseException as exc:
        failure=repr(exc)
        raise
    finally:
        if proc is not None and proc.poll() is None:code=stop_owned(proc)
        receipt={'exit_code':code,'job':a.job,'branch':a.branch,'release_identity':identity,
                 'deadline_unix':end,'injected_crash':a.inject_crash,'log_stem':stamp,'production':production,'failure':failure}
        c.write(logs/(stamp+'.json'),receipt)
        if production:
            approval=read(ctx.output/'launch-approval.json');approval.update(approved=False,closed_utc=c.utc())
            c.write(ctx.output/'launch-approval.json',approval)
    print(json.dumps(receipt));raise SystemExit(code)
if __name__=='__main__':main()
