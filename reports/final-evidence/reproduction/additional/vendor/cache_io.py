"""Disk-backed normal features with bounded, byte-preserving batch reads."""
import ctypes,gc,hashlib,os
from pathlib import Path
import numpy as np
import torch

class Counters(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong)]+[(k,ctypes.c_size_t) for k in
        ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage',
         'QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]

def private_bytes():
    c=Counters();c.cb=ctypes.sizeof(c)
    process=ctypes.windll.kernel32.GetCurrentProcess
    process.restype=ctypes.c_void_p
    fn=ctypes.windll.psapi.GetProcessMemoryInfo
    fn.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    if not fn(process(),ctypes.byref(c),ctypes.sizeof(c)):raise ctypes.WinError()
    return c.PrivateUsage

class BatchStore:
    def __init__(self,path):self.path=Path(path)
    def __getitem__(self,index):
        mm=np.load(self.path,mmap_mode='r',allow_pickle=False)
        try:
            assert mm.dtype==np.float32
            array=mm[index.numpy()].copy(order='C')
        finally:mm._mmap.close()
        return torch.from_numpy(array)

def make_features(model,rows,namespace,root):
    guard=namespace['guard'];resources=namespace['resources'];images=namespace['images']
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    files=[];hashers=[];shapes=[];payload=0
    with torch.inference_mode():
        for offset in range(0,len(rows),4):
            guard();values=model.feature_extractor(images(rows[offset:offset+4]))
            if not files:
                shapes=[(len(rows),*v.shape[1:]) for v in values]
                payload=sum(int(np.prod(s))*4 for s in shapes)
                if payload>2*2**30:raise RuntimeError('Feature-cache ceiling')
                if resources()['disk_free']<payload+8*2**30:raise RuntimeError('Feature-cache disk headroom')
                batch_bytes=sum(v.numel()*4 for v in values)
                if resources()['ram_available']<batch_bytes*3+int(1.25*2**30):raise RuntimeError('Feature-cache batch RAM headroom')
                for level,shape in enumerate(shapes):
                    path=root/f'level-{level}.npy'
                    mm=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=shape)
                    mm.flush();mm._mmap.close();files.append(path);hashers.append(hashlib.sha256())
            for path,v,h in zip(files,values,hashers,strict=True):
                arr=np.ascontiguousarray(v.cpu().numpy())
                h.update(memoryview(arr).cast('B'))
                mm=np.load(path,mmap_mode='r+',allow_pickle=False)
                try:mm[offset:offset+len(arr)]=arr;mm.flush()
                finally:mm._mmap.close()
                del arr,mm
        del values
    return [BatchStore(p) for p in files],{'logical_payload_bytes':payload,'shapes':[list(s) for s in shapes],
        'feature_payload_sha256':[h.hexdigest() for h in hashers],'cache_files':[str(p) for p in files]}

def wrapper(original,namespace,mode,folder):
    def features(model,rows):
        gc.collect();before=private_bytes()
        if mode=='mapped':
            cache,record=make_features(model,rows,namespace,folder/'caches'/f'process-{os.getpid()}')
        else:
            cache=original(model,rows)
            record={'logical_payload_bytes':sum(v.numel()*v.element_size() for v in cache),
                'shapes':[list(v.shape) for v in cache],
                'feature_payload_sha256':[hashlib.sha256(memoryview(v.numpy()).cast('B')).hexdigest() for v in cache]}
        gc.collect();after=private_bytes()
        record.update(mode=mode,private_before=before,private_after=after,private_increase=after-before,
            normal_rows=len(rows),pid=os.getpid())
        namespace['write'](folder/f'cache-memory-{os.getpid()}.json',record)
        return cache
    return features
