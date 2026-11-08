"""Sequential Dinosaur seeds: gate torque calibration, train, evaluate and render.

Run with the repository virtualenv. State/logs persist under --out-root; this
runner deliberately fails visibly on calibration or subprocess failure.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--out-root', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--stage-iterations', type=int, default=0, help='Evaluate this milestone before continuing to final budget')
    parser.add_argument('--iterations', type=int, default=200000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.out_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = args.calibration.resolve()
    launch = json.loads((base/'launch.json').read_text())
    config = launch['config']
    source = (launch['command'][launch['command'].index('--wm-init') + 1]
              if '--wm-init' in launch['command'] else None)
    state = dict(worker_pid=os.getpid(), seeds=args.seeds, target_iteration=args.iterations)
    env = dict(os.environ, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONPATH=str(root/'src'))

    def save(**fields):
        state.update(fields)
        path = output/'status.tmp'
        path.write_text(json.dumps(state, indent=2)+'\n')
        path.replace(output/'status.json')

    def command(cfg, out, mode, checkpoint):
        cmd = [sys.executable, '-u', '-m', 'blinkamp.train', '--log-dir', str(out)]
        if mode is not None:
            cmd += [mode, str(checkpoint)]
        for key, value in cfg.items():
            flag = '--'+key.replace('_', '-')
            cmd += [flag if value else '--no-'+key.replace('_', '-')] if isinstance(value, bool) else [flag, str(value)]
        return cmd

    def run(cmd, out, log):
        with (out/log).open('a') as stream:
            child = subprocess.Popen(cmd, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT)
            save(child_pid=child.pid, command=cmd)
            code = child.wait()
        if code:
            raise RuntimeError(f'{log}: subprocess exited {code}')

    def gate(cal):
        rows = list(csv.DictReader((cal/'train_log.csv').open()))
        row = rows[-1]
        actual, demand, zero_a, zero_d = [float(row[k]) for k in (
            'torque_probe_actual_rel_mae', 'torque_probe_demand_rel_mae',
            'torque_probe_zero_actual_rel_mae', 'torque_probe_zero_demand_rel_mae')]
        passed = (int(row['iteration']) == 500 and int(row['nonfinite']) == 0
                  and all(math.isfinite(x) for x in (actual,demand,zero_a,zero_d))
                  and actual < min(.1,zero_a) and demand < min(.1,zero_d)
                  and not row['actor_loss'] and (cal/'ckpt_000500.pt').exists())
        result = dict(passed=passed, iteration=500, policy_updates=0,
                      actual_mae_over_limit=actual, demand_mae_over_limit=demand,
                      zero_actual_mae_over_limit=zero_a, zero_demand_mae_over_limit=zero_d,
                      relative_mae_limit=.1, validation='reserved environments excluded from WM training')
        (cal/'calibration_acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
        if not passed:
            raise RuntimeError(f'Calibration rejected: {result}')

    try:
        for seed in args.seeds:
            cal = base if seed == config['seed'] else output/f'seed{seed}_calibration'
            cfg = dict(config, seed=seed)
            save(status='calibrating', seed=seed, run=str(cal))
            if cal == base:
                # Existing seed0 calibration must finish before invoking the runner.
                gate(cal)
            else:
                cal.mkdir(exist_ok=False)
                run(command(cfg,cal,'--wm-init' if source else None,source),cal,'stdout.log')
                gate(cal)
            out = output/f'seed{seed}_200k'
            out.mkdir(exist_ok=False)
            shutil.copy2(cal/'train_log.csv',out/'train_log.csv')
            shutil.copytree(cal/'tensorboard',out/'tensorboard')
            stages = ([args.stage_iterations] if 500 < args.stage_iterations < args.iterations else []) + [args.iterations]
            resume = cal/'ckpt_000500.pt'
            for target in stages:
                cfg.update(iterations=target,ckpt_every=5000,log_every=50)
                cmd = command(cfg,out,'--resume',resume)
                (out/f'launch_{target:06d}.json').write_text(json.dumps(dict(config=cfg,command=cmd,calibration=str(cal)),indent=2)+'\n')
                save(status='training',run=str(out),stage_target=target)
                run(cmd,out,'stdout.log')
                final = out/f'ckpt_{target:06d}.pt'
                if not final.exists():
                    raise RuntimeError('Missing final checkpoint')
                rows = list(csv.DictReader((out/'train_log.csv').open()))
                if int(rows[-1]['iteration']) != target or int(rows[-1]['nonfinite']) != 0:
                    raise RuntimeError('Training stage failed iteration/nonfinite acceptance')
                suffix = 'final' if target == args.iterations else f'{target:06d}'
                save(status='evaluating')
                run([sys.executable,'-u','-m','blinkamp.evaluate',str(final),
                     '--out',str(out/f'eval_{suffix}.json'),'--record',str(out/f'rollout_{suffix}.npz')],out,f'eval_{suffix}_stdout.log')
                save(status='rendering')
                run([sys.executable,str(root/'tools/render_amp.py'),str(out/f'rollout_{suffix}.npz'),
                     str(out/f'rollout_{suffix}.mp4')],out,f'render_{suffix}_stdout.log')
                resume = final
            save(last_completed_seed=seed)
        save(status='complete',finished_unix=time.time())
    except BaseException:
        save(status='failed',error=traceback.format_exc(),finished_unix=time.time())
        raise


if __name__ == '__main__':
    main()
