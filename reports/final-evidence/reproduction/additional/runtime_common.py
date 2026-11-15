"""Operational common-module adapter; scientific worker and model files are unchanged."""
import ctypes,datetime,hashlib,json,os,shutil,sys,time
from pathlib import Path
from configuration import read,sha,digest as _plain_digest

_CTX=None
_FREEZE=None
_GATE=None
HERE=None
REPO=None

def configure(ctx):
    global _CTX,_FREEZE,HERE,REPO
    _CTX=ctx;HERE=ctx.output;REPO=ctx.package
    _FREEZE=ctx.verify_sources()
    os.environ.update(MPLBACKEND='Agg',HF_HUB_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',
        CUBLAS_WORKSPACE_CONFIG=':4096:8',MPLCONFIGDIR=str(HERE/'runtime-cache/mpl'),
        TORCH_HOME=str(HERE/'runtime-cache/torch'),PYTHONDONTWRITEBYTECODE='1')

def digest(value):return _plain_digest(_CTX.canonical(value))
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def key(row):return digest({'path':row['path']})[:24]
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    temp.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temp.replace(path)

class Memory(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ('total','available','total_page','available_page','total_virtual','available_virtual','extended')]
def resources():
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise RuntimeError('RAM query failed')
    return {'ram_available':m.available,'disk_free':shutil.disk_usage(HERE).free}

def conditions():
    for p in [HERE/'PAUSE',_CTX.scratch/'PAUSE',*map(Path,_CTX.host.get('pause_paths',[]))]:
        if p.exists():raise RuntimeError('Pause marker: '+str(p))
    deadline=os.environ.get('STUDY_DEADLINE_UNIX')
    if not deadline or time.time()>=float(deadline):raise RuntimeError('Bounded deadline reached')

def guard():
    if _GATE is None:raise RuntimeError('Resource gate must be configured')
    return _GATE.check()
def validate():
    if _CTX.verify_sources()!=_FREEZE:raise RuntimeError('Release identity changed')
    return _FREEZE
def lock_gpu():
    import msvcrt
    _CTX.gpu_lock.parent.mkdir(parents=True,exist_ok=True)
    handle=_CTX.gpu_lock.open('a+b');handle.seek(0)
    try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:handle.close();raise RuntimeError('Another configured GPU worker owns the lock')
    return handle
def stop_owned_worker(proc):
    if proc.poll() is None:proc.kill()
    return proc.wait(timeout=10)
def access_guard(roots,rows):
    roots=[os.path.normcase(str(Path(r).resolve()))+os.sep for r in roots]
    allowed={os.path.normcase(str(Path(r['path']).resolve())) for r in rows};opened=set()
    def audit(event,args):
        if event not in ('open','os.listdir','os.scandir') or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=os.path.normcase(os.path.abspath(os.fsdecode(args[0])))
        if any(p.startswith(root) or p==root.rstrip(os.sep) for root in roots):
            if event!='open' or p not in allowed:raise PermissionError('FORBIDDEN_DATA_ACCESS: '+p)
            opened.add(p)
    sys.addaudithook(audit);return opened
def verify_row(row):
    if 'sha256' not in row or sha(row['path'])!=row['sha256']:raise RuntimeError('Input checksum changed')
