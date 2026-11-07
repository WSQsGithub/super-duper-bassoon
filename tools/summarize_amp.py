"""Collect completed pilot evaluations without hiding missing/failed experiments."""

import argparse
import csv
import json
from pathlib import Path


def summarize(root):
    experiments = []
    for name in ('taskonly', 'walk', 'highknees', 'dinosaur'):
        run = root / name
        entry = {'experiment': name, 'complete': False}
        if (run / 'args.json').exists():
            entry['config'] = json.loads((run / 'args.json').read_text())
        if (run / 'train_log.csv').exists():
            rows = list(csv.DictReader((run / 'train_log.csv').open()))
            if rows:
                last = rows[-1]
                entry.update(iteration=int(last['iteration']), training_minutes=float(last['wall_seconds']) / 60,
                             nonfinite_updates=int(last['nonfinite']), real_task=float(last['real_task']),
                             wm_validation_root_m=float(last['wm_probe_root_m']))
        if (run / 'eval.json').exists():
            evaluation = json.loads((run / 'eval.json').read_text())
            entry['evaluation'] = evaluation
            forward = [x for x in evaluation['results'] if x['command'][0] > 0]
            entry['passed_forward_targets'] = bool(forward) and all(
                x['survival_rate'] >= .95 and x['velocity_rmse_m_s'] is not None
                and x['velocity_rmse_m_s'] <= .15 for x in forward)
            entry['complete'] = evaluation['iteration'] == entry.get('config', {}).get('iterations')
        experiments.append(entry)
    return {'root': str(root), 'complete': all(e['complete'] for e in experiments),
            'experiments': experiments, 'scope': 'Single-seed exploratory pilots; not a convergence or speed claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    summary = summarize(args.root)
    (args.root / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    for e in summary['experiments']:
        print(f"{e['experiment']}: iteration={e.get('iteration', 0)}, eval_complete={e['complete']}, "
              f"forward_targets={e.get('passed_forward_targets', 'pending')}")


if __name__ == '__main__':
    main()
