"""Run the authorized B-only pilot, evaluating immutable checkpoints as they appear.
The previous multi-seed queue and watchdog remain stopped. No automatic extension.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=Path('configs/blinkamp/style_B.json'))
    parser.add_argument('--calibration',type=Path,default=Path('logs/blinkamp/original_budget_calibration_mlp/ckpt_000500.pt'))
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=False)
    config=args.config.resolve();cfg=json.loads(config.read_text());cal=args.calibration.resolve()
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(root/'src'))
    state=dict(status='starting',worker_pid=os.getpid(),group='B',seed=cfg['seed'],target_iteration=10500,
        target_actor_update_rounds=10000,calibration=str(cal),calibration_sha256=hashlib.sha256(cal.read_bytes()).hexdigest(),
        config=cfg,evaluated=[])
    def save(**fields):
        state.update(fields,updated_unix=time.time());temp=out/'status.tmp'
        temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(out/'status.json')
    def run(cmd,logname):
        with (out/logname).open('a') as log:
            p=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            save(evaluation_pid=p.pid)
            if p.wait():raise RuntimeError(f'{logname}: failed')
        save(evaluation_pid=None)
    train=[sys.executable,'-u','-m','blinkamp.train','--config',str(config),'--log-dir',str(out),
           '--resume',str(cal),'--fork-command-profile']
    probe_commands=[[.4,0.,0.],[-.27,0.,0.],[.08,-.22,0.],[.08,-.14,0.]]
    (out/'probe_commands.json').write_text(json.dumps(probe_commands,indent=2)+'\n')
    state['probe_note']='Near-expert speed probes; last slow point is absent as a nonzero command under A deadband. Not a strict common-support certificate.'
    (out/'launch.json').write_text(json.dumps(dict(command=train,**state),indent=2)+'\n')
    process=None
    try:
        with (out/'stdout.log').open('a') as log:
            process=subprocess.Popen(train,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        save(status='training',training_pid=process.pid)
        milestones=[1500,3500,4000,10000,10500]
        for target in milestones:
            ck=out/f'ckpt_{target:06d}.pt'
            while not ck.exists():
                if process.poll() is not None:
                    raise RuntimeError(f'Training exited {process.returncode} before checkpoint {target}')
                time.sleep(5)
            save(status='evaluating',evaluation_iteration=target,actor_update_rounds=target-500)
            dest=out/f'eval_{target:06d}';dest.mkdir(exist_ok=True)
            record=dest/'model.npz'
            run([sys.executable,'-u','-m','blinkamp.evaluate',str(ck),'--record-all','--record',str(record),
                 '--out',str(dest/'broad.json')],f'eval_{target:06d}.log')
            run([sys.executable,'-u','-m','blinkamp.evaluate',str(ck),'--commands-json',str(out/'probe_commands.json'),
                 '--record-all','--record',str(dest/'near_expert.npz'),'--out',str(dest/'near_expert.json')],f'probe_{target:06d}.log')
            for command_id in (1,2,4):
                run([sys.executable,str(root/'tools/render_amp.py'),str(dest/f'model_command_{command_id:02d}.npz'),
                     str(dest/f'model_command_{command_id:02d}.mp4'),'--root-view'],f'render_{target:06d}.log')
            # Historical A has equal-budget checkpoints at 3500/10000 (3000/9500 actor rounds).
            baseline=root/f'logs/blinkamp/original_budget_suite_mlp/seed0_200k/ckpt_{target:06d}.pt'
            if target in (3500,10000) and baseline.exists():
                run([sys.executable,'-u','-m','blinkamp.evaluate',str(baseline),'--record-all',
                     '--record',str(dest/'A_model.npz'),'--out',str(dest/'A_broad.json')],f'A_eval_{target:06d}.log')
                run([sys.executable,'-u','-m','blinkamp.evaluate',str(baseline),'--commands-json',str(out/'probe_commands.json'),
                     '--record-all','--record',str(dest/'A_near_expert.npz'),'--out',str(dest/'A_near_expert.json')],f'A_probe_{target:06d}.log')
            state['evaluated'].append(target);save(status='training')
        if process.wait():raise RuntimeError('Training exited unsuccessfully')
        save(status='complete',finished_unix=time.time())
    except BaseException:
        if process is not None and process.poll() is None:
            # Evaluation failure must not silently leave an unmonitored training process.
            process.terminate();process.wait()
        save(status='failed',error=traceback.format_exc(),finished_unix=time.time())
        raise

if __name__=='__main__':main()
