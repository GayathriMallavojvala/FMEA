"""Run: python evaluate_stochastic.py --replicates 20 --output results/stochastic"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import platform
from pathlib import Path

from benchmark.simulation import PROFILES, models, sample_episode, validate_scenario
from benchmark.evaluation import STRATEGIES, run_episode
from benchmark.statistics import summarize, paired_comparisons


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def report(path, summaries, paired, replicates):
    lines = ['# Stochastic benchmark pilot', '',
        'Synthetic sensitivity analysis only. No real plant cases or trained models.', '',
        f'{replicates} independent draws per cause per scenario/profile, with all causes covered.',
        'Metrics reweight the stratified draws to the stated environment prior; scenarios receive equal weight.',
        'Intervals resample the 10 authored scenario groups (2.5th–97.5th percentiles). They do not establish industrial generalization.', '',
        '| Profile | Method | Accuracy | Mean tests | Mean total loss (Rs) | Confident-error rate |',
        '|---|---|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['profile']} | {s['strategy']} | {s['correct']:.1%} | {s['tests']:.2f} | {s['total_loss_inr']:.0f} | {s['confident_error']:.1%} |")
    lines += ['', '## Paired monetary VOI minus information-gain/cost', '',
        'Negative loss difference favors monetary VOI; intervals crossing zero do not establish a consistent advantage across these templates.', '',
        '| Profile | Mean total-loss difference (Rs) | Scenario-bootstrap 95% interval |',
        '|---|---:|---:|']
    for row in paired:
        if row['comparator'] == 'cost_aware' and row['metric'] == 'total_loss_inr':
            lines.append(f"| {row['profile']} | {row['difference']:.0f} | [{row['ci_low']:.0f}, {row['ci_high']:.0f}] |")
    lines += ['', '## Interpretation limits', '',
        '- No uncertainty-aware proposed policy or escalation action is implemented yet.',
        '- All current methods always output a candidate cause. A high-belief rate is not abstention coverage.',
        '- Unknown-cause episodes cannot be correctly classified by this closed candidate set. Their failures are a deliberate stress test.',
        '- Unknown outcomes are synthetic uniform draws, not observations of real omitted causes.',
        '- Correlation uses shared inverse-CDF uniforms; strength and sign depend on outcome ordering. This is a sensitivity model, not a calibrated joint plant model.',
        '- Inconclusive tests consume their full cost and cannot be repeated; belief stays unchanged. Policies do not anticipate missingness.',
        '- Likelihood mismatch, missingness, correlations and unknown prevalence are authored intervention settings.',
        '- Costs are summed serial test time; parallel testing and action-dependent consequences are outside this benchmark.',
        '- Brier scores are unnormalized multiclass scores in [0,2]. Unknown truth is an additional class assigned probability zero; known-only scores are also reported.',
        '- Confidence intervals cover variation among these authored templates; template count remains 10.',
        '- No CAPA text quality, citation accuracy, external LLM or live UI evaluation is included.',
        '- This is a pilot. More random draws reduce simulation noise but cannot replace independently sourced incidents.']
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replicates', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20261029)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--profiles', nargs='+', choices=list(PROFILES), default=list(PROFILES))
    parser.add_argument('--loss-inr', type=float, default=20000)
    parser.add_argument('--time-weight', type=float, default=20)
    parser.add_argument('--threshold', type=float, default=0.85)
    parser.add_argument('--output', type=Path, default=Path('results/stochastic'))
    args = parser.parse_args()
    if args.replicates < 1 or args.bootstrap < 1 or len(set(args.profiles)) != len(args.profiles):
        parser.error('Positive replicates/bootstrap and unique profiles required')
    if not all(math.isfinite(v) and v >= 0 for v in [args.loss_inr, args.time_weight]) or not 0 < args.threshold <= 1:
        parser.error('Invalid loss, time weight or threshold')
    root = Path(__file__).resolve().parent
    source = root/'active_diagnosis/diagnosis_library.json'
    scenarios = json.loads(source.read_text())['scenarios']
    for sc in scenarios:
        validate_scenario(sc)
    if len({sc['id'] for sc in scenarios}) != len(scenarios):
        parser.error('Duplicate scenario IDs')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Output directory must be empty; use a new path to preserve prior runs')
    output.mkdir(parents=True, exist_ok=True)
    rows, model_records = [], []
    episodes_count = 0
    with gzip.open(output/'episodes.jsonl.gz', 'wt', encoding='utf-8') as episodes, gzip.open(output/'traces.jsonl.gz', 'wt', encoding='utf-8') as traces:
        for profile in args.profiles:
            for sc in scenarios:
                policy, environment = models(sc, profile)
                model_records.append(dict(policy=policy, environment=environment))
                for cause in environment['prior']:
                    for replicate in range(args.replicates):
                        episode = sample_episode(environment, cause, replicate, args.seed, args.replicates)
                        episodes.write(json.dumps(episode, sort_keys=True)+'\n')
                        episodes_count += 1
                        for strategy in STRATEGIES:
                            row, trace = run_episode(policy, episode, strategy, args.seed, args.loss_inr, args.time_weight, args.threshold)
                            rows.append(row)
                            # Engine's hidden-label correctness field is intentionally discarded.
                            trace.pop('correct', None)
                            traces.write(json.dumps(dict(episode_id=episode['episode_id'], strategy=strategy, **trace), sort_keys=True)+'\n')
            print(f'Completed {profile}: {len(rows)} policy runs so far', flush=True)
    summaries, calibration = summarize(rows, args.seed, args.bootstrap)
    paired = paired_comparisons(rows, args.seed, args.bootstrap)
    for name, records in [('episodes_metrics.csv', rows), ('summary.csv', summaries), ('calibration.csv', calibration), ('paired_comparisons.csv', paired)]:
        write_csv(output/name, records)
    (output/'models.json').write_text(json.dumps(model_records, indent=2))
    report(output/'REPORT.md', summaries, paired, args.replicates)
    files = [source, root/'active_diagnosis/engine.py', Path(__file__).resolve(), *sorted((root/'benchmark').glob('*.py'))]
    manifest = dict(kind='synthetic stochastic pilot; not industrial validation',
        seed=args.seed, replicates_per_cause=args.replicates, independent_scenarios=len(scenarios),
        profiles={key: PROFILES[key] for key in args.profiles}, strategies=STRATEGIES,
        episodes=episodes_count, policy_runs=len(rows), loss_inr=args.loss_inr,
        time_weight_inr_per_min=args.time_weight, threshold=args.threshold,
        bootstrap_draws=args.bootstrap, bootstrap_unit='scenario',
        sampling='stratified by cause; scenario-weighted by environment prior',
        python=platform.python_version(),
        code_and_data_sha256={str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'Saved {episodes_count} paired episodes and {len(rows)} policy runs in {output}')

if __name__ == '__main__':
    main()
