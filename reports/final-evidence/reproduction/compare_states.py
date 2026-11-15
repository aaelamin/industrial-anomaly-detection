"""Short-lived CPU-only exact comparison; no dataset access or model inference."""
import argparse,sys
from pathlib import Path
from suite_common import read,sha,write,equal
def main():
    p=argparse.ArgumentParser();p.add_argument('--left',required=True);p.add_argument('--right',required=True)
    p.add_argument('--state',required=True,choices=['checkpoint.pt','resume.pt','memory.pt']);p.add_argument('--out',required=True)
    a=p.parse_args();import torch
    left,right=Path(a.left),Path(a.right)
    x=torch.load(left/a.state,map_location='cpu',weights_only=True);y=torch.load(right/a.state,map_location='cpu',weights_only=True)
    assert equal(x,y),'Model/optimizer/RNG or bank state differs'
    del x,y
    x=torch.load(left/'normal-probe.pt',map_location='cpu',weights_only=True);y=torch.load(right/'normal-probe.pt',map_location='cpu',weights_only=True)
    assert equal(x,y),'Normal probes differ'
    write(a.out,{'status':'PASS_BITWISE_STATE_AND_NORMAL_PROBE','bitwise_state':True,'bitwise_probe':True,
        'state_file':a.state,'left_state_sha256':sha(left/a.state),'right_state_sha256':sha(right/a.state),
        'source_dataset_images_opened':0})
    print('Exact state and normal-probe equality passed')
if __name__=='__main__':main()
