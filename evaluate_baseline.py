"""Reproducible deterministic logic check, NOT a publication benchmark."""
import csv
import hashlib
import json
from pathlib import Path
from active_diagnosis import engine


def main():
    root = Path(__file__).resolve().parent
    source = root / 'active_diagnosis/diagnosis_library.json'
    scenarios = json.loads(source.read_text())['scenarios']
    output = root / 'results'
    output.mkdir(exist_ok=True)
    rows = []
    for scenario in scenarios:
        for strategy in ['cost_aware_voi', 'cost_aware', 'greedy_info', 'cheapest_first', 'fixed_order', 'random']:
            result = engine.run_diagnosis(scenario, strategy, loss_inr=20000)
            rows.append(dict(scenario=scenario['id'], strategy=strategy,
                correct=result['correct'], model_belief=result['final_confidence'],
                tests=result['num_tests_used'], direct_cost_inr=result['total_cost_inr'],
                time_minutes=result['total_time_minutes'],
                total_loss_inr=result['total_cost_inr'] + 20*result['total_time_minutes'] + 20000*(not result['correct']),
                test_sequence=';'.join(result['tests_used'])))
    with (output/'baseline.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output/'manifest.json').write_text(json.dumps(dict(
        kind='deterministic logic check; not empirical validation',
        library_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        prior='scenario authored prior', loss_inr=20000,
        time_cost_inr_per_min=20, confidence_threshold=engine.CONFIDENCE_THRESHOLD,
        outcome='most likely given configured true cause',
        random_order_seed='CRC32 of scenario ID', episodes=len(rows)), indent=2))
    print(f'Wrote {len(rows)} deterministic baseline traces to {output}')

if __name__ == '__main__':
    main()
