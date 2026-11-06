"""Audit stored pairing, metrics, traces, and exact policy replay for a saved run."""
import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path
from benchmark.evaluation import run_episode, score_brier
from benchmark.simulation import INCONCLUSIVE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root/'manifest.json').read_text())
    model_records = json.loads((root/'models.json').read_text())
    models = {(r['environment']['profile'], r['environment']['scenario_id']): r['policy'] for r in model_records}
    with gzip.open(root/'episodes.jsonl.gz', 'rt') as f:
        episodes = {r['episode_id']: r for r in map(json.loads, f)}
    with (root/'episodes_metrics.csv').open() as f:
        rows = {(r['episode_id'], r['strategy']): r for r in csv.DictReader(f)}
    assert len(episodes) == manifest['episodes']
    assert len(rows) == manifest['policy_runs']
    weights = defaultdict(float)
    replayed = set()
    count = 0
    with gzip.open(root/'traces.jsonl.gz', 'rt') as f:
        for trace in map(json.loads, f):
            count += 1
            episode = episodes[trace['episode_id']]
            row = rows[(trace['episode_id'], trace['strategy'])]
            policy = models[episode['profile'], episode['scenario_id']]
            tests = {t['id']: t for t in policy['tests']}
            selected = trace['tests_used']
            assert len(selected) == len(set(selected)) == int(row['tests'])
            assert selected == [step['test_id'] for step in trace['steps']]
            assert len(selected) == trace['num_tests_used']
            for step in trace['steps']:
                assert step['outcome'] == episode['outcomes'][step['test_id']]
                assert math.isclose(sum(step['belief_after'].values()), 1)
                if step['outcome'] == INCONCLUSIVE:
                    assert step['belief_before'] == step['belief_after']
            cost = sum(tests[tid]['cost_inr'] for tid in selected)
            duration = sum(tests[tid]['time_minutes'] for tid in selected)
            correct = trace['final_diagnosis'] == episode['true_cause']
            loss = cost + manifest['time_weight_inr_per_min']*duration + manifest['loss_inr']*(not correct)
            assert math.isclose(loss, float(row['total_loss_inr']))
            assert correct == bool(int(row['correct']))
            assert math.isclose(score_brier(trace['final_belief'], episode['true_cause']), float(row['brier']))
            assert math.isclose(max(trace['final_belief'].values()), float(row['model_belief']))
            weights[episode['profile'], episode['scenario_id'], trace['strategy']] += float(row['scenario_weight'])
            group = episode['profile'], trace['strategy']
            if group not in replayed:
                expected_row, expected_trace = run_episode(policy, episode, trace['strategy'], manifest['seed'], manifest['loss_inr'], manifest['time_weight_inr_per_min'], manifest['threshold'])
                expected_trace.pop('correct', None)
                assert trace == dict(episode_id=episode['episode_id'], strategy=trace['strategy'], **expected_trace)
                for key, value in expected_row.items():
                    if isinstance(value, (int, float)):
                        assert math.isclose(float(row[key]), value)
                    else:
                        assert row[key] == value
                replayed.add(group)
    assert count == len(rows)
    assert all(math.isclose(weight, 1) for weight in weights.values())
    for eid in episodes:
        assert all((eid, strategy) in rows for strategy in manifest['strategies'])
    print(f'PASS: {len(episodes)} complete paired episodes, {count} audited traces, {len(replayed)} exact profile/method replays; weights and metrics verified.')

if __name__ == '__main__':
    main()
