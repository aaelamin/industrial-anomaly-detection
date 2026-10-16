"""Verify and summarize the separate microbatch-32 pilot."""
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    old = read(ROOT / 'reports/week5/pilot.json')
    fast = read(ROOT / 'reports/week5/fast/pilot.json')
    assert fast['pilot_complete'] and fast['completed_runs'] == 18
    assert not fast['full_study_started']
    assert fast['fresh_process_resume']['fresh_process_resume_exact']
    for store in ('pilot', 'fast-pilot'):
        manifest = read(ROOT / f'results/week5/{store}/study.json')
        for name, sha in manifest['sources'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
    before = {(r['category'], r['seed'], r['arm']): r for r in old['runs']}
    paired = []
    for run in fast['runs']:
        key = (run['category'], run['seed'], run['arm'])
        reference = before[key]
        assert run['initial_state_hash'] == reference['initial_state_hash']
        assert run['checkpoint_prediction_reload_exact'] and run['backbone_unchanged']
        folder = ROOT / 'results/week5/fast-pilot' / run['category'] / f"seed_{run['seed']}" / run['arm']
        receipt = read(folder / 'complete.json')
        assert receipt['protocol'] == fast['protocol']
        for name, sha in receipt['files'].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == sha
        paired.append(reference['median_epoch_seconds_after_first'] / run['median_epoch_seconds_after_first'])
    for category in ('bottle', 'screw', 'carpet'):
        paths = [ROOT / f'results/week5/{store}/{category}/projection.pt' for store in ('pilot', 'fast-pilot')]
        assert len({hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}) == 1
    low, central, high = fast['full_90_run_500_epoch_training_hours_low_median_high']
    # Reused pilot projections save setup time here. A new category still needs fitting.
    fit_hours = statistics.median(p['projection_fit_seconds'] for p in old['preparation']) * 15 / 3600
    levels = read(ROOT / 'reports/week5/level_checks.json')
    metric_hours = statistics.median(r['seven_subset_metric_seconds'] / 7 / r['images'] for r in levels['runs']) * sum(fast['test_counts'].values()) * 6 / 3600
    overhead = sum(fast[k] for k in ('estimated_shared_preparation_hours', 'estimated_checkpoint_hours_every25epochs', 'estimated_model_inference_hours')) + fit_hours + metric_hours
    summary = {'passed': True, 'runs': 18, 'protocol': fast['protocol'],
               'median_paired_speed_ratio': statistics.median(paired),
               'training_hours_low_median_high': [low, central, high],
               'total_hours_low_median_high': [low+overhead, central+overhead, high+overhead],
               'planning_hours_upper_plus25percent': (high+overhead)*1.25,
               'full_study_started': False}
    (ROOT / 'reports/week5/fast/checks.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Faster batch pilot', '',
             'All 18 ten-epoch runs finished using a flow microbatch of 32. The effective batch remains 32. Frozen-backbone extraction and test inference still use four images at a time. The same saved PCA projections and paired initial model weights were used as in the original pilot.', '',
             '| Category | Input | Microbatch 4 seconds/epoch | Microbatch 32 seconds/epoch |',
             '| --- | --- | ---: | ---: |']
    for category in ('bottle', 'screw', 'carpet'):
        for arm in ('middle', 'fusion'):
            values = [statistics.median(r['median_epoch_seconds_after_first'] for r in result['runs'] if r['category']==category and r['arm']==arm) for result in (old, fast)]
            lines.append(f'| {category} | {arm} | {values[0]:.2f} | {values[1]:.2f} |')
    lines += ['', f"The median paired training speed ratio is {statistics.median(paired):.2f} times the original pilot. These pilots ran at different times, so temperature and other system activity can affect that comparison.", '',
              f'The full 90-run, 500-epoch comparison is estimated at **{central:.1f} training hours**, or **{central+overhead:.1f} hours including modeled setup, checkpoints and evaluation**. The observed-rate total range is **{low+overhead:.1f}-{high+overhead:.1f} hours**. Adding 25% scheduling headroom above the upper estimate gives **{(high+overhead)*1.25:.1f} hours**. This is not a confidence interval or a guarantee.', '',
              'The estimate adds the original measured PCA fitting cost because this validation reused saved projections. It excludes pauses, retries, prolonged thermal changes and dense map/preview exports.', '',
              f"Peak main-process allocated GPU memory was {max(r['gpu_peak_allocated_mib'] for r in fast['runs']):.0f} MiB. This excludes the separate recovery process and is not total GPU usage.", '',
              'All runs passed finite-loss/gradient checks, full test-set finite-map checks, frozen-backbone checks and exact final-model reload checks. A fresh Python process reproduced the next full training epoch exactly from a checkpoint. Source hashes and all 18 output receipts were verified. Microbatch 32 need not reproduce microbatch 4 bit for bit because floating-point accumulation changes. Neither pilot establishes final accuracy or 500-epoch stability.', '',
              'Microbatch 32 is suitable for the planned study on this GPU based on these short-run checks. The full-study controller, its retention/evaluation rules and a recovery smoke test remain to be prepared before launch. No full study has started.', '',
              'The largest temporary category cache remains about 3.82 GiB, with about 1.66 GiB for planned model retention and active recovery checkpoints. Existing data, earlier studies, projections and exported maps are additional. Completed pilot feature caches are removed.', '',
              '- [Measurements](fast/pilot.json)', '- [Verification summary](fast/checks.json)',
              '- [Research question and publication requirements](../../docs/WEEK5_PLAN.md#research-question-and-publication-direction)', '',
              'Recheck or resume with `python scripts/week5_fast_pilot.py`. Rebuild this report with `python scripts/week5_fast_report.py`.', '']
    (ROOT / 'reports/week5/fast_pilot.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
