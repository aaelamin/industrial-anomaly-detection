"""Delete only exact files registered as this overlay's temporary scratch."""
import json,os,re,time,hashlib
from pathlib import Path

def save(p,value):
    tmp=p.with_name(p.name+'.partial');tmp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
class Scratch:
    def __init__(self,root,job):
        if not re.fullmatch(r'[a-z0-9_-]+',job):raise ValueError('Invalid scratch job')
        self.root=Path(root).resolve();self.job=job;self.scope=(self.root/job).resolve()
        assert self.scope.is_relative_to(self.root) and self.scope!=self.root
        self.scope.mkdir(parents=True,exist_ok=True)
    def create(self):
        import psutil
        pid=os.getpid();folder=self.scope/f'worker-{pid}-{time.time_ns()}'
        folder.mkdir(exist_ok=False)
        expected=[folder/'caches'/f'process-{pid}'/f'level-{k}.npy' for k in range(3)]
        save(folder/'owner.json',{'type':'temporary_normal_features_v1','job':self.job,'pid':pid,
            'process_created':psutil.Process(pid).create_time(),'folder':str(folder),
            'files':[str(p) for p in expected],'cleaned':False})
        return folder
    def clean(self,folder,current=False):
        import psutil
        folder=Path(folder).resolve()
        if folder.parent!=self.scope:raise ValueError('Scratch cleanup outside exact job scope')
        p=folder/'owner.json';owner=json.loads(p.read_text(encoding='utf-8'))
        assert owner['type']=='temporary_normal_features_v1' and owner['job']==self.job
        assert Path(owner['folder']).resolve()==folder
        pid=owner['pid'];expected=[folder/'caches'/f'process-{pid}'/f'level-{k}.npy' for k in range(3)]
        if [str(x) for x in expected]!=owner['files']:raise ValueError('Registered paths differ from the exact owned files')
        for f in expected:
            if not f.resolve().is_relative_to(self.scope):raise ValueError('Scratch path escaped scope')
        if owner['cleaned']:return
        try:alive=psutil.Process(pid).create_time()==owner['process_created']
        except psutil.NoSuchProcess:alive=False
        if alive and not (current and pid==os.getpid()):raise RuntimeError('Refuse cleanup of a live owner')
        deleted=[]
        for f in expected:
            if not f.exists():continue
            record={'path':str(f),'bytes':f.stat().st_size,'sha256':sha(f)}
            # Individual, prevalidated owned files only; never recursive deletion.
            until=time.monotonic()+2
            while True:
                try:f.unlink();break
                except PermissionError:
                    if time.monotonic()>=until:raise
                    time.sleep(.05)
            deleted.append(record)
        save(folder/'cleanup.json',{'deleted_temporary_files':deleted,'other_files_untouched':True})
        owner['cleaned']=True;save(p,owner)
    def recover(self):
        for p in sorted(self.scope.glob('worker-*/owner.json')):self.clean(p.parent)
