"""Scenario-weighted summaries and paired scenario-cluster bootstrap intervals."""
from collections import defaultdict
import random
from statistics import mean
from benchmark.simulation import stable_seed

METRICS = ['correct', 'tests', 'direct_cost_inr', 'time_minutes',
           'effective_test_cost_inr', 'decision_loss_inr', 'total_loss_inr',
           'brier', 'confident', 'confident_error', 'inconclusive_tests']


def weighted(rows, key):
    total = sum(row['scenario_weight'] for row in rows)
    return sum(row['scenario_weight']*row[key] for row in rows)/total if total else None


def bootstrap(values, seed, draws=2000):
    if not values:
        raise ValueError('No scenario estimates')
    if draws < 1:
        raise ValueError('Bootstrap draws must be positive')
    rng = random.Random(seed)
    samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return samples[int(0.025*(draws-1))], samples[int(0.975*(draws-1))]


def summarize(rows, seed, draws=2000):
    groups = defaultdict(list)
    for row in rows:
        groups[row['profile'], row['strategy']].append(row)
    summaries, calibration = [], []
    for (profile, strategy), group in sorted(groups.items()):
        scenarios = defaultdict(list)
        for row in group:
            scenarios[row['scenario_id']].append(row)
        summary = dict(profile=profile, strategy=strategy, episodes=len(group), scenarios=len(scenarios))
        for metric in METRICS:
            values = [weighted(scenarios[sid], metric) for sid in sorted(scenarios)]
            summary[metric] = mean(values)
            if metric in ['total_loss_inr', 'correct', 'brier', 'confident_error']:
                low, high = bootstrap(values, stable_seed(seed, profile, strategy, metric), draws)
                summary[metric+'_ci_low'], summary[metric+'_ci_high'] = low, high
        confident = [row for row in group if row['confident']]
        known = [row for row in group if row['known_cause']]
        unknown = [row for row in group if not row['known_cause']]
        summary['error_given_confident'] = (1-weighted(confident, 'correct')) if confident else None
        summary['known_cause_brier'] = weighted(known, 'brier')
        summary['unknown_confident_rate'] = weighted(unknown, 'confident')
        summary['unknown_accuracy'] = weighted(unknown, 'correct')
        ece = 0
        total_weight = sum(r['scenario_weight'] for r in group)
        for bin_index in range(10):
            selected = [r for r in group if min(int(r['model_belief']*10), 9) == bin_index]
            mass = sum(r['scenario_weight'] for r in selected)/total_weight
            confidence, accuracy = weighted(selected, 'model_belief'), weighted(selected, 'correct')
            if selected:
                ece += mass*abs(confidence-accuracy)
            calibration.append(dict(profile=profile, strategy=strategy,
                bin_lower=bin_index/10, bin_upper=(bin_index+1)/10,
                count=len(selected), weight_mass=mass, mean_confidence=confidence, accuracy=accuracy))
        summary['ece_10_bins'] = ece
        summaries.append(summary)
    return summaries, calibration


def paired_comparisons(rows, seed, draws=2000, reference='cost_aware_voi'):
    indexed = {}
    for row in rows:
        key = row['profile'], row['episode_id'], row['strategy']
        if key in indexed:
            raise ValueError('Duplicate episode/strategy result')
        indexed[key] = row
    comparisons = []
    profiles = sorted({r['profile'] for r in rows})
    strategies = sorted({r['strategy'] for r in rows} - {reference})
    for profile in profiles:
        references = [r for r in rows if r['profile'] == profile and r['strategy'] == reference]
        for strategy in strategies:
            differences = defaultdict(list)
            for ref in references:
                other = indexed[(profile, ref['episode_id'], strategy)]
                if ref['scenario_weight'] != other['scenario_weight']:
                    raise ValueError('Unpaired weights')
                item = dict(scenario_weight=ref['scenario_weight'])
                for metric in ['total_loss_inr', 'correct', 'tests', 'confident_error']:
                    item[metric] = ref[metric]-other[metric]
                differences[ref['scenario_id']].append(item)
            for metric in ['total_loss_inr', 'correct', 'tests', 'confident_error']:
                values = [weighted(differences[sid], metric) for sid in sorted(differences)]
                low, high = bootstrap(values, stable_seed(seed, profile, strategy, metric, 'paired'), draws)
                comparisons.append(dict(profile=profile, reference=reference, comparator=strategy,
                    metric=metric, difference=mean(values), ci_low=low, ci_high=high,
                    paired_episodes=len(references), scenarios=len(values)))
    return comparisons
