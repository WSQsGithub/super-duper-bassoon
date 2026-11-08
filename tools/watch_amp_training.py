"""Watch the authorized AMP suite, recover crashes and extend unconverged runs.

No training hyperparameters are changed. State and every decision are persisted
inside the suite. A live orphan training child is allowed to finish before recovery.
"""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def load_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        return default


def rows_at(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [r for r in csv.DictReader(stream) if str(r.get('iteration','')).isdigit() and str(r.get('nonfinite','')).isdigit()]


def alive(pid, marker):
    if not pid:
        return False
    try:
        path = Path(f'/proc/{pid}')
        return marker.encode() in (path/'cmdline').read_bytes() and (path/'stat').read_text().split()[2] != 'Z'
    except (OSError,IndexError):
        return False


def continuation_reasons(report, rows, window=20000):
    reasons = []
    result = report.get('results',[])
    if not result:
        return ['missing_evaluation']
    for r in result:
        if r['survival_rate'] < 1.:
            reasons.append('survival_below_100_percent')
        for key in ('velocity_rmse_m_s','yaw_rmse_rad_s'):
            value = r.get(key)
            if value is None or not math.isfinite(value) or value > .2:
                reasons.append(key + '_above_0.2')
    end = int(report['iteration'])
    recent = [r for r in rows if end-window < int(r['iteration']) <= end]
    previous = [r for r in rows if end-2*window < int(r['iteration']) <= end-window]
    if not recent or not previous:
        reasons.append('insufficient_trend_history')
    else:
        for key in ('velocity_error','yaw_error','real_reward'):
            first = sum(float(r[key]) for r in previous)/len(previous)
            last = sum(float(r[key]) for r in recent)/len(recent)
            if not math.isfinite(first+last) or abs(last-first)/max(abs(first),.01) >= .01:
                reasons.append(key + '_still_changing')
    return sorted(set(reasons))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--poll-seconds',type=float,default=30.)
    parser.add_argument('--extend-by',type=int,default=50000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    suite = args.suite.resolve()
    # flock prevents multiple watchdogs starting duplicate recovery jobs.
    lock = (suite/'watchdog.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state = load_json(suite/'watchdog_status.json',{})
    state.update(pid=os.getpid(),status='monitoring',started_unix=time.time())
    pending = state.get('pending',{})
    failures = state.get('failures',0)

    def save(**fields):
        state.update(fields,updated_unix=time.time(),pending=pending,failures=failures)
        temp = suite/'watchdog_status.tmp'
        temp.write_text(json.dumps(state,indent=2)+'\n')
        temp.replace(suite/'watchdog_status.json')

    def event(**data):
        with (suite/'watchdog_events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(),**data))+'\n')

    def launch(seeds,target,allowance):
        cmd = [sys.executable,'-u',str(root/'tools/run_amp_suite.py'),
               '--calibration',str(args.calibration.resolve()),'--out-root',str(suite),
               '--continue-existing','--accepted-nonfinite',str(allowance),
               '--seeds',*[str(s) for s in seeds],'--stage-iterations','3500',
               '--iterations',str(target),'--pi-microbatch','0']
        with (suite/'watchdog_children.log').open('a') as log:
            process = subprocess.Popen(cmd,cwd=root,stdin=subprocess.DEVNULL,stdout=log,
                stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(root/'src')))
        pending.update(pid=process.pid,seeds=seeds,target=target)
        event(action='launch',pid=process.pid,seeds=seeds,target=target,accepted_nonfinite=allowance,command=cmd)
        save(status='monitoring',last_launch_unix=time.time())

    save()
    while True:
        try:
            current = load_json(suite/'status.json',{})
            live = (alive(pending.get('pid'),'run_amp_suite.py') or
                    alive(current.get('worker_pid'),'run_amp_suite.py') or
                    alive(current.get('child_pid'),'blinkamp.train') or
                    alive(current.get('child_pid'),'blinkamp.evaluate') or
                    alive(current.get('child_pid'),'render_amp.py'))
            if live:
                save(status='monitoring',suite_status=current.get('status'),active_seed=current.get('seed'))
                time.sleep(args.poll_seconds)
                continue
            if current.get('status') != 'complete':
                # Retry backoff; repeated deterministic failures stop with evidence.
                signature = (current.get('error','unknown'),current.get('stage_target'))
                old = state.get('failure_signature')
                failures = failures+1 if old == list(signature) else 1
                state['failure_signature'] = list(signature)
                if failures > 3:
                    save(status='needs_attention',reason='same failure persisted through three recoveries')
                    return
                seeds = current.get('seeds') or [0,1]
                target = current.get('target_iteration',200000)
                allowance = 1
                for seed in seeds:
                    rows = rows_at(suite/f'seed{seed}_200k/train_log.csv')
                    if rows:
                        count = int(rows[-1]['nonfinite'])
                        if count > 1:
                            end = int(rows[-1]['iteration'])
                            tail = [r for r in rows if int(r['iteration']) >= end-1000]
                            if count > 5 or end < 2000 or not tail or any(int(r['nonfinite']) != count for r in tail):
                                save(status='needs_attention',reason='nonfinite updates recur; no automatic relaxation')
                                return
                            allowance = max(allowance,count)
                event(action='recover',suite_error=current.get('error'),accepted_nonfinite=allowance)
                launch(seeds,target,allowance)
                time.sleep(args.poll_seconds)
                continue
            pending.clear()
            failures = 0
            extensions = []
            for seed in (0,1):
                out = suite/f'seed{seed}_200k'
                report = load_json(out/'eval_final.json',{})
                reasons = continuation_reasons(report,rows_at(out/'train_log.csv'))
                if reasons:
                    extensions.append((seed,report.get('iteration',200000)+args.extend_by,reasons))
            if not extensions:
                save(status='complete',reason='both seeds meet operational tracking and curve-stability criteria; style still requires visual acceptance')
                return
            # Prefer the seed with lower completed budget to avoid starving seed1.
            seed,target,reasons = min(extensions,key=lambda x:x[1])
            rows = rows_at(suite/f'seed{seed}_200k/train_log.csv')
            allowance = max(1,int(rows[-1]['nonfinite']))
            event(action='extend',seed=seed,target=target,reasons=reasons)
            save(status='extending',extension_reasons=reasons)
            launch([seed],target,allowance)
            time.sleep(args.poll_seconds)
        except Exception:
            event(action='watchdog_error',error=traceback.format_exc())
            save(status='retrying_watchdog_error')
            time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
