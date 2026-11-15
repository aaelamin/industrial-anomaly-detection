"""Path, source and process controls for the configured reproduction suite."""
import datetime, hashlib, importlib.metadata, json, os, shutil, subprocess, sys, time
from pathlib import Path

PACKAGE=Path(__file__).resolve().parent
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def digest(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+'.partial');t.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');t.replace(p)
def absolute(v):
    p=Path(v)
    if not p.is_absolute(): raise ValueError('Host paths must be absolute')
    return p.resolve()
def disjoint(values):
    pairs=list(values.items())
    for i,(n,p) in enumerate(pairs):
        for m,q in pairs[i+1:]:
            if p.is_relative_to(q) or q.is_relative_to(p): raise ValueError('Overlapping roots: '+n+', '+m)

class Suite:
    def __init__(self,config):
        self.config=Path(config).resolve();self.host=read(self.config)
        self.workspace=absolute(self.host['workspace_root'])
        self.roots={k:absolute(v) for k,v in self.host['data_roots'].items()}
        assert set(self.roots)=={'mvtec','aebad'}
        disjoint({'package':PACKAGE,'workspace':self.workspace,**self.roots})
        self.gpu_lock=absolute(self.host['gpu_lock'])
        if any(self.gpu_lock.is_relative_to(p) for p in self.roots.values()): raise ValueError('GPU lock is in dataset')
        self.torch_home=absolute(self.host['torch_home'])
        self.weights={k:absolute(v) for k,v in self.host['weight_paths'].items()}
        self.manifest=read(PACKAGE/'suite-manifest.json')
        self.data=read(PACKAGE/'additional/manifest.json')
        if set(self.weights)!=set(self.data['weights']):raise ValueError('Provide exactly the declared weight identities')
        self.project=self.workspace/'legacy'
        self.pause_paths=[absolute(p) for p in self.host.get('pause_paths',[])]
        self.extra=[absolute(p) for p in self.host.get('extra_pythonpath',[])]
        for p in reversed(self.extra):
            if not p.is_dir():raise ValueError('Missing explicitly configured dependency directory')
            if str(p) not in sys.path:sys.path.insert(0,str(p))
    def identity(self):
        return digest({'suite':read(PACKAGE/'suite-lock.json'),'manifest':sha(PACKAGE/'suite-manifest.json'),
                       'additional':read(PACKAGE/'additional/release-lock.json')})
    def check_sources(self,packages=True):
        for name,h in self.manifest['legacy_sources'].items():
            p=PACKAGE/'legacy'/name
            assert p.resolve().is_relative_to(PACKAGE/'legacy') and sha(p)==h,name
        for name,h in read(PACKAGE/'suite-lock.json')['files'].items():
            p=(PACKAGE/name).resolve()
            assert p.is_relative_to(PACKAGE) and sha(p)==h,name
        sys.path.insert(0,str(PACKAGE/'additional'))
        from configuration import Configuration
        for name,h in self.data['source_files'].items(): assert sha(PACKAGE/'additional'/name)==h,name
        for name,h in read(PACKAGE/'additional/release-lock.json')['files'].items(): assert sha(PACKAGE/'additional'/name)==h,name
        if packages:
            for name,version in self.data['packages'].items(): assert importlib.metadata.version(name)==version,name
        return self.identity()
    def conditions(self,end):
        if time.time()>=end: raise RuntimeError('Suite wall cap reached')
        for p in self.pause_paths:
            if p.exists(): raise RuntimeError('Pause marker: '+str(p))
    def env(self,end):
        env=os.environ.copy()
        env.update(TORCH_HOME=str(self.torch_home),MPLCONFIGDIR=str(self.workspace/'runtime-cache/mpl'),
                   HF_HOME=str(self.workspace/'runtime-cache/hf'),HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
                   MPLBACKEND='Agg',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONDONTWRITEBYTECODE='1',
                   STUDY_DEADLINE_UNIX=str(end-15))
        if self.extra:env['PYTHONPATH']=os.pathsep.join(map(str,self.extra))
        return env
    def materialize(self):
        self.workspace.mkdir(parents=True,exist_ok=True)
        marker=self.workspace/'suite-identity.json'
        if marker.exists(): assert read(marker)['identity']==self.identity(),'Different suite identity; use a new workspace'
        elif any(self.workspace.iterdir()): raise ValueError('Refusing nonempty unidentified workspace')
        else:write(marker,{'identity':self.identity(),'utc':utc()})
        for name,h in self.manifest['legacy_sources'].items():
            dest=self.project/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists(): assert sha(dest)==h,name
            else:shutil.copy2(PACKAGE/'legacy'/name,dest)
        link=self.project/'data/mvtec_ad';link.parent.mkdir(parents=True,exist_ok=True)
        if link.exists():assert link.resolve()==self.roots['mvtec'],'Different dataset junction'
        else:
            if os.name!='nt': raise RuntimeError('This qualified source route is Windows-specific')
            quote=lambda p:"'"+str(p).replace("'","''")+"'"
            command='New-Item -ItemType Junction -Path '+quote(link)+' -Target '+quote(self.roots['mvtec'])+' | Out-Null'
            result=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],capture_output=True,text=True)
            if result.returncode:raise RuntimeError('Dataset junction setup failed: '+result.stderr.strip())
            assert link.resolve()==self.roots['mvtec']
        config={'data_roots':{k:str(v) for k,v in self.roots.items()},'weight_paths':{k:str(v) for k,v in self.weights.items()},
                'output_root':str(self.workspace/'additional-output'),'scratch_root':str(self.workspace/'additional-scratch'),
                'gpu_lock':str(self.gpu_lock),'pause_paths':list(map(str,self.pause_paths))}
        target=self.workspace/'additional-host.json'
        if target.exists():assert read(target)==config
        else:write(target,config)
    def verify_weights(self):
        for h,p in self.weights.items(): assert p.is_file() and sha(p)==h,'Weight checksum mismatch'
        for name in ('week3','week4'):
            cfg=read(PACKAGE/f'legacy/configs/{name}.json')
            p=self.torch_home/'hub/checkpoints'/Path(cfg['weights_url']).name
            assert p.is_file() and sha(p)==cfg['weights_sha256'],'Missing exact legacy weight: '+str(p)
    def lock_gpu(self):
        import msvcrt
        self.gpu_lock.parent.mkdir(parents=True,exist_ok=True)
        h=self.gpu_lock.open('a+b');h.seek(0)
        try:msvcrt.locking(h.fileno(),msvcrt.LK_NBLCK,1)
        except BaseException:h.close();raise
        return h
    def command(self,argv,label,end,expected=0):
        # Fail before spawning a child if owned-process cleanup is unavailable.
        import psutil
        self.conditions(end);logs=self.workspace/'suite-logs';logs.mkdir(exist_ok=True)
        stamp=str(time.time_ns())+'-'+label;proc=None;failure=None;code=None
        record={'command':list(map(str,argv)),'started_utc':utc(),'deadline_unix':end,'expected_exit':expected}
        try:
            with (logs/(stamp+'.out')).open('x') as out,(logs/(stamp+'.err')).open('x') as err:
                proc=subprocess.Popen(list(map(str,argv)),cwd=self.project,env=self.env(end),stdout=out,stderr=err,
                                      creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                while proc.poll() is None:
                    self.conditions(end)
                    if shutil.disk_usage(self.workspace).free<5*2**30:raise RuntimeError('Suite disk floor reached')
                    if os.name=='nt':
                        import ctypes
                        class Memory(ctypes.Structure):
                            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total','available','total_page','available_page','total_virtual','available_virtual','extended']]
                        m=Memory();m.length=ctypes.sizeof(m)
                        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise RuntimeError('Memory query failed')
                        if m.available<.75*2**30:raise RuntimeError('Suite critical RAM floor reached')
                    time.sleep(1)
                code=proc.returncode
            if code!=expected:raise RuntimeError(f'{label} exited {code}; see {logs/(stamp+".err")}')
        except BaseException as exc:failure=repr(exc);raise
        finally:
            if proc is not None and proc.poll() is None:
                try:children=psutil.Process(proc.pid).children(recursive=True)
                except psutil.NoSuchProcess:children=[]
                for child in reversed(children):
                    try:child.kill()
                    except psutil.NoSuchProcess:pass
                psutil.wait_procs(children,timeout=5)
                if proc.poll() is None:proc.kill()
                code=proc.wait(timeout=10)
            record.update(exit_code=code,finished_utc=utc(),failure=failure)
            write(logs/(stamp+'.json'),record)
        return record

def equal(a,b):
    import torch
    if type(a)!=type(b):return False
    if isinstance(a,torch.Tensor):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
