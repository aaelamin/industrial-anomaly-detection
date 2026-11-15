"""Verify the actual pre-commit state in a short-lived CPU process."""
import argparse
from pathlib import Path
from suite_common import read,write,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--kind',choices=['week3','flow','patchcore'],required=True);p.add_argument('--out',required=True)
    a=p.parse_args();folder=Path(a.folder);marker=read(folder/'injected-crash.json')
    result={'kind':a.kind,'source_images_opened':0,'gpu_used':False}
    if a.kind=='patchcore':
        assert marker['before_memory_commit'] and not (folder/'memory.pt').exists()
        result['committed_bank_absent']=True
    else:
        import torch
        path=folder/('checkpoint.pt' if a.kind=='week3' else 'resume.pt')
        state=torch.load(path,map_location='cpu',weights_only=True)
        assert state['epoch']==1,'Interrupted branch must still have the epoch-one checkpoint'
        result.update(checkpoint_epoch=1,state_sha256=sha(path))
    result['status']='PASS_ACTUAL_PRECOMMIT_STATE';write(a.out,result);print(result['status'])
if __name__=='__main__':main()
