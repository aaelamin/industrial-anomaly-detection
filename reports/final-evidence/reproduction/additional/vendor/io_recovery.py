"""Bounded retries for the frozen runner's same-directory .partial commits only."""
import os,time,json,threading
from pathlib import Path
ORIGINAL_REPLACE=Path.replace
HERE=Path(__file__).resolve().parent
STUDY=HERE.parent/'full-study-20261103'

def event(value):
    folder=HERE/'io-events';folder.mkdir(exist_ok=True)
    with (folder/f'{time.time_ns()}-{os.getpid()}.json').open('x',encoding='utf-8') as f:
        json.dump({'unix':time.time(),'pid':os.getpid(),**value},f)

def install(timeout=5.0,inject=False):
    roots=[STUDY.resolve(),HERE.resolve()];injected=set()
    def replace(self,target):
        src=Path(self).absolute();dst=Path(target).absolute()
        eligible=(str(src)==str(dst)+'.partial' and any(dst.is_relative_to(root) for root in roots))
        if not eligible:return ORIGINAL_REPLACE(self,target)
        if inject and dst.name in ('progress.json','resume.pt') and dst.exists() and dst.name not in injected:
            injected.add(dst.name);held=dst.open('rb')
            threading.Thread(target=lambda:(time.sleep(.3),held.close()),daemon=True).start()
            event({'kind':'injected_reader_lock','target':str(dst),'hold_seconds':.3})
        started=time.monotonic();failures=0
        while True:
            try:result=ORIGINAL_REPLACE(self,target)
            except PermissionError as exc:
                if getattr(exc,'winerror',None) not in (5,32):raise
                failures+=1
                if time.monotonic()-started>=timeout:
                    event({'kind':'retry_exhausted','target':str(dst),'failures':failures,'elapsed':time.monotonic()-started,'winerror':exc.winerror})
                    raise
                time.sleep(min(.1,max(0,timeout-(time.monotonic()-started))))
            else:
                if failures:event({'kind':'recovered','target':str(dst),'failures':failures,'elapsed':time.monotonic()-started})
                return result
    Path.replace=replace
    return replace
