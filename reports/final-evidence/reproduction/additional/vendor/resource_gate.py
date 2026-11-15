"""Allocation-free bounded waits at existing resource checkpoints."""
import os,time
from contextlib import contextmanager

GIB=2**30
class ResourceGate:
    def __init__(self,resources,check,log=lambda event:None,clock=time.monotonic,sleep=time.sleep):
        self.resources=resources;self.conditions=check;self.log=log;self.clock=clock;self.sleep=sleep
        self.minimum=GIB;self.recovery=1.25*GIB;self.cap=120;self.stage='runtime'
        self.inject_remaining=0
    @contextmanager
    def allocation(self,minimum,recovery=None,cap=120,stage='allocation'):
        old=(self.minimum,self.recovery,self.cap,self.stage)
        self.minimum=minimum;self.recovery=minimum+.25*GIB if recovery is None else recovery;self.cap=cap;self.stage=stage
        try:yield self
        finally:self.minimum,self.recovery,self.cap,self.stage=old
    def check(self):
        started=None;stable=0;next_log=0
        while True:
            self.conditions();r=dict(self.resources());injected=False
            if self.stage=='runtime' and self.inject_remaining:
                r['ram_available']=min(r['ram_available'],int(.95*GIB));self.inject_remaining-=1;injected=True
            now=self.clock()
            if r['disk_free']<5*GIB:raise RuntimeError(f'Resource floor: disk {r}')
            if r['ram_available']<.5*GIB:raise RuntimeError(f'Resource floor: critical RAM {r}')
            if started is None and r['ram_available']>=self.minimum:return r
            if started is None:
                started=now;self.log({'event':'wait_started','stage':self.stage,'resources':r,'injected':injected})
            if r['ram_available']>=self.recovery:stable+=1
            else:stable=0
            if stable>=3:
                self.log({'event':'wait_recovered','stage':self.stage,'seconds':now-started,'resources':r});return r
            if now-started>=self.cap:raise RuntimeError(f'Resource floor: bounded RAM wait expired ({self.stage}, {self.cap}s)')
            if now>=next_log:
                self.log({'event':'waiting','stage':self.stage,'seconds':now-started,'resources':r});next_log=now+10
            self.sleep(min(2,self.cap-(now-started)))

def conditions(common):
    def check():
        paths=[common.HERE/'PAUSE',common.HERE.parent/'PAUSE',common.HERE.parent/'PAUSE.md',common.REPO/'PAUSE',common.REPO/'work/PAUSE',common.HERE.parent/'work-block-20261107/standardization-screen/PAUSE']
        for p in paths:
            if p.exists():raise RuntimeError(f'Pause marker: {p}')
        deadline=os.environ.get('STUDY_DEADLINE_UNIX')
        if deadline and time.time()>=float(deadline):raise RuntimeError('Frozen time cap reached')
    return check
