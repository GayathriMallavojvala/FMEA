"""Replay one saved episode without resampling or changing its evidence."""
import argparse
import gzip
import json
from pathlib import Path
from benchmark.evaluation import STRATEGIES, run_episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('episode_id')
    parser.add_argument('--strategy', choices=STRATEGIES, default='cost_aware_voi')
    args = parser.parse_args()
    manifest = json.loads((args.directory/'manifest.json').read_text())
    records = json.loads((args.directory/'models.json').read_text())
    episode = None
    with gzip.open(args.directory/'episodes.jsonl.gz', 'rt') as source:
        for line in source:
            candidate = json.loads(line)
            if candidate['episode_id'] == args.episode_id:
                episode = candidate
                break
    if episode is None:
        parser.error('Episode ID not present in this run')
    model = next(record['policy'] for record in records
                 if record['environment']['profile'] == episode['profile']
                 and record['environment']['scenario_id'] == episode['scenario_id'])
    row, trace = run_episode(model, episode, args.strategy, manifest['seed'],
                            manifest['loss_inr'], manifest['time_weight_inr_per_min'], manifest['threshold'])
    trace.pop('correct', None)
    print(json.dumps(dict(metrics=row, trace=trace), indent=2))

if __name__ == '__main__':
    main()
