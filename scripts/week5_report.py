"""Summarize the measured pilot cost without launching training."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/week5'


def main():
    report=json.loads((OUT/'pilot.json').read_text())
    tests=json.loads((OUT/'tests.json').read_text())
    assert report['pilot_complete'] and report['completed_runs']==18 and not report['full_study_started']
    assert tests['passed'] and report['fresh_process_resume']['fresh_process_resume_exact']
    runs=report['runs']
    for run in runs:
        folder=ROOT/'results/week5/pilot'/run['category']/f"seed_{run['seed']}"/run['arm']
        receipt=json.loads((folder/'complete.json').read_text())
        assert receipt['protocol']==report['protocol']
        for name,sha in receipt['files'].items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==sha
    with (OUT/'pilot_runs.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=runs[0].keys())
        writer.writeheader()
        writer.writerows(runs)
    low,median,high=report['full_90_run_500_epoch_training_hours_low_median_high']
    overhead=sum(report[k] for k in ['estimated_shared_preparation_hours',
        'estimated_checkpoint_hours_every25epochs','estimated_model_inference_hours'])
    metric_hours=None
    if (OUT/'level_checks.json').exists():
        levels=json.loads((OUT/'level_checks.json').read_text())
        if levels['complete']:
            seconds_per_image=statistics.median(r['seven_subset_metric_seconds']/7/r['images'] for r in levels['runs'])
            metric_hours=seconds_per_image*sum(report['test_counts'].values())*6/3600
            overhead+=metric_hours
    lines=['# Week 5 pilot','','Both feature inputs use the same 96-channel flow architecture. The middle-only control uses 96 projected middle-level channels. Fusion uses 32 projected channels from each of three levels. PCA and standardization were fitted on sampled normal training features only.', '',
        f"Each arm has {runs[0]['trainable_parameters']:,} trainable parameters. The pretrained feature extractor and fitted PCA projections stay frozen.", '',
        'The pilot ran both arms for ten epochs on bottle, screw and carpet with seeds 0, 1 and 2. These 18 runs check operation and cost. Their short-run accuracy was not used to choose settings. No full training study has been started.', '',
        '## Measured training time','','The table uses the median of epochs 2-10 for each run, then the median across its three seeds. Preparation, evaluation and checkpoint writes are timed separately.', '',
        '| Category | Training images | Steps per epoch | Middle seconds per epoch | Fusion seconds per epoch |',
        '| --- | ---: | ---: | ---: | ---: |']
    for category in ('bottle','screw','carpet'):
        selected=[r for r in runs if r['category']==category]
        values=[statistics.median(r['median_epoch_seconds_after_first'] for r in selected if r['arm']==arm) for arm in ('middle','fusion')]
        lines.append(f"| {category} | {selected[0]['train_images']} | {selected[0]['steps_per_epoch']} | {values[0]:.2f} | {values[1]:.2f} |")
    lines+=['','## Estimate for the full comparison','',
        f'At 500 epochs, two arms, 15 categories and three seeds, the full comparison would contain 90 runs. Scaling the measured time per optimizer step by the actual training counts gives **{median:.1f} hours of training**, with **{low:.1f}-{high:.1f} hours** from the fastest and slowest observed pilot rates.', '',
        f"Shared feature extraction and projection fitting: about {report['estimated_shared_preparation_hours']:.2f} hours. Checkpoint writes every 25 epochs: about {report['estimated_checkpoint_hours_every25epochs']:.2f} hours. Model inference on the test images: about {report['estimated_model_inference_hours']:.2f} hours.", '']
    if metric_hours is not None:
        lines += [f'The feature-level inspection also measured metric computation. Extrapolating its time per image gives roughly {metric_hours:.2f} additional hours for image AUROC, pixel AUROC and pixel AP across the full comparison. This uses different predicted maps and test group sizes, so it is approximate.', '']
    lines += [f'The modeled total is **{median+overhead:.1f} hours**, with an observed-rate range of **{low+overhead:.1f}-{high+overhead:.1f} hours**. Allowing 25% planning headroom above the upper estimate gives **{1.25*(high+overhead):.1f} hours**. That allowance is a scheduling choice, not a measured confidence bound.', '',
        'The estimate assumes category feature caches are shared across seeds and arms. It excludes pauses, retries, long-run thermal changes, preview/map exports and most process startup. A ten-epoch pilot cannot establish 500-epoch stability or final accuracy.', '',
        '## Memory and storage','',
        f"The largest measured allocated GPU memory was {max(r['gpu_peak_allocated_mib'] for r in runs):.0f} MiB. This is PyTorch allocation in the main process, not total system or GPU usage, and excludes the separate recovery-check process.", '',
        f"The largest category's native and projected training caches are estimated at {report['estimated_largest_category_temporary_feature_gib']:.2f} GiB. They are temporary and should be processed one category at a time. Keeping four late/final models per run plus two active recovery checkpoints is estimated at {report['estimated_full_model_and_recovery_gib']:.2f} GiB. Dataset storage, existing studies, projections and exported test maps are additional.", '',
        'The completed pilot retains projections, final models and the latest two recovery checkpoints for each run. Its completed category feature caches were removed. Old studies and their data were left unchanged.', '',
        '## Checks','',
        f"All {tests['tests']} targeted tests passed. They check projection algebra and feature routing, rejection of test images during fitting, constant-feature handling, gradient accumulation, and continuation with reconstructed training objects.", '',
        'A separate Python process loaded the bottle control checkpoint and reproduced the next full epoch exactly, including model weights, optimizer state, random states and training diagnostics. All 18 final models also reproduced their saved-model probe predictions exactly after reload. The frozen backbone remained unchanged, and each seed pair had identical initial downstream weights and trainable parameter counts.', '',
        'PCA sample identifiers, training-image hashes and scale-floor counts are saved with each projection. Disk-cached projected features matched a fresh extraction exactly. Full pilot test-set inference produced finite maps with the expected shape. This checks operation, not final quality.', '',
        '## Next step','',
        'Review the measured budget before starting a full comparison. The inference-only [feature-level results](levels.md) help interpret the existing models, but should not be used to silently change the pilot design or select a different best subset for each category. The feature-fusion proposal remains an experiment rather than an established improvement.', '',
        '- [Per-run timing and memory](pilot_runs.csv)', '- [Detailed benchmark and preparation times](pilot.json)',
        '- [Implementation and experiment plan](../../docs/WEEK5_PLAN.md)', '',
        'Run `python scripts/week5_pilot.py` to check or resume this fixed ten-epoch pilot. Finished runs are checksum-verified and skipped. Run `python scripts/week5_levels.py` for the saved-model level inspection, then `python scripts/week5_report.py` to rebuild this report. There is no full-study launch option in the pilot command.', '']
    if (OUT/'batch_probe.json').exists():
        probe=json.loads((OUT/'batch_probe.json').read_text())
        assert probe['complete']
        lines += ['## Extra batch-size timing check', '',
            'We also timed microbatches of 4, 8, 16 and 32 while keeping the effective batch at 32. Each setting repeated 32 normal bottle images for 12 optimizer steps, excluding the first three from timing. This is a throughput check, not another accuracy experiment.', '',
            '| Input | Microbatch | Seconds per step | Peak allocated MiB |',
            '| --- | ---: | ---: | ---: |']
        for row in probe['runs']:
            if 'error' in row:
                lines.append(f"| {row['arm']} | {row['micro_batch_size']} | {row['error']} | |")
            else:
                lines.append(f"| {row['arm']} | {row['micro_batch_size']} | {row['steady_seconds_per_step']:.3f} | {row['gpu_peak_allocated_mib']:.0f} |")
        lines += ['', 'The main estimate above remains based on the 18-run microbatch-4 pilot. A faster setting here would still need its own longer pilot and recovery check before changing the study configuration.', '', '- [Batch-size timing measurements](batch_probe.json)', '']
    if (OUT/'fast/checks.json').exists() and json.loads((OUT/'fast/checks.json').read_text())['passed']:
        lines[1:1] = ['', 'This is the original microbatch-4 pilot. The later [microbatch-32 validation](fast_pilot.md) passed all 18 runs and provides the revised time estimate.']
    (OUT/'pilot.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Pilot report: {OUT / "pilot.md"}')


if __name__=='__main__':
    main()
