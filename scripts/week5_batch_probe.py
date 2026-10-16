"""Measure larger microbatches on 32 normal bottle images after the pilot."""
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.week5_model import ProjectedFlow, Projection, train_epoch, seed_all, torch, np
from scripts.week4_colab import write_json
from scripts.week4_preflight import DiskFeatures
from week3 import build_model, MVTecDataset, DataLoader, file_hash


def main():
    report=json.loads((ROOT/'reports/week5/pilot.json').read_text())
    assert report['pilot_complete']
    config=json.loads((ROOT/'configs/week5.json').read_text())
    seed_all(0)
    backbone=build_model(json.loads((ROOT/config['backbone_config']).read_text()),'cuda').feature_extractor.eval()
    projection_path=ROOT/'results/week5/pilot/bottle/projection.pt'
    receipt=json.loads(projection_path.with_suffix('.json').read_text())
    assert file_hash(projection_path)==receipt['sha256']
    projection=Projection(torch.load(projection_path,map_location='cpu',weights_only=True)).cuda()
    dataset=MVTecDataset(ROOT/'data/mvtec_ad','bottle','train',256)
    paths=[p.relative_to(dataset.root).as_posix() for p in dataset.paths[:32]]
    scratch=ROOT/'work/week5-batch-probe'
    scratch.mkdir(parents=True,exist_ok=True)
    caches={arm:[] for arm in config['arms']}
    with torch.inference_mode():
        for start in range(0,32,4):
            images=torch.stack([dataset[i][0] for i in range(start,start+4)]).cuda()
            native=backbone(images)
            for arm in config['arms']:
                caches[arm].append(projection(native,arm).cpu().numpy())
    for arm,parts in caches.items():
        np.save(scratch/f'{arm}.npy',np.concatenate(parts))
    del caches,native,images,projection
    results=[]
    for arm in config['arms']:
        features=DiskFeatures(scratch/f'{arm}.npy')
        for micro in (4,8,16,32):
            seed_all(0)
            chosen={**config,'micro_batch_size':micro}
            model=ProjectedFlow(chosen).cuda()
            opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                lr=chosen['learning_rate'],weight_decay=chosen['weight_decay'])
            generator=torch.Generator().manual_seed(0)
            torch.cuda.reset_peak_memory_stats()
            times=[]
            diagnostics=[]
            try:
                for step in range(12):
                    torch.cuda.synchronize()
                    tick=time.perf_counter()
                    diagnostics.append(train_epoch(model,features,opt,generator,chosen))
                    torch.cuda.synchronize()
                    times.append(time.perf_counter()-tick)
                row={'arm':arm,'micro_batch_size':micro,'effective_batch_size':32,
                    'steady_seconds_per_step':statistics.median(times[3:]),
                    'gpu_peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                    'gpu_peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
                    'step_seconds':times,'finite_losses_and_gradients':True}
            except torch.OutOfMemoryError:
                row={'arm':arm,'micro_batch_size':micro,'error':'CUDA out of memory'}
            results.append(row)
            write_json(ROOT/'reports/week5/batch_probe.json',{'complete':False,'runs':results})
            print(row,flush=True)
            del model,opt
            torch.cuda.empty_cache()
        features.values._mmap.close()
        (scratch/f'{arm}.npy').unlink()
    scratch.rmdir()
    write_json(ROOT/'reports/week5/batch_probe.json',{'complete':True,'runs':results,
        'normal_training_images':paths,'source_sha256':file_hash(__file__),
        'limits':'Throughput probe on 32 repeated normal images. Each setting uses 12 optimizer steps, with the first three excluded from timing. This does not replace the 18-run microbatch-4 pilot, test accuracy, or validate ten-epoch/fresh-process recovery at the larger microbatch. It does not change the saved experiment config.'})


if __name__=='__main__':
    main()
