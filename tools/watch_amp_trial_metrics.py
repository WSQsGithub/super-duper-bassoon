"""Read-only observer: compute identical pose metrics for B and historical A.
Does not launch, stop, resume or modify training.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',type=Path,required=True)
    args=parser.parse_args();trial=args.trial.resolve();root=Path(__file__).resolve().parents[1]
    while True:
        summary=[]
        for directory in sorted(trial.glob('eval_*')):
            if not directory.is_dir():continue
            item=dict(iteration=int(directory.name.split('_')[-1]))
            for label,report_name,prefix in [('B','broad.json','model'),('A','A_broad.json','A_model')]:
                report_file=directory/report_name;dest=directory/f'{label}_style.json'
                if not report_file.exists():continue
                report=json.loads(report_file.read_text())
                if not dest.exists():
                    with (directory/f'{label}_style.log').open('w') as log:
                        result=subprocess.run([sys.executable,str(root/'tools/analyze_amp_command_support.py'),
                            '--checkpoint',report['checkpoint'],'--record-root',str(directory),'--record-prefix',prefix,
                            '--out',str(dest)],cwd=root,env=dict(os.environ,OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode:
                        item[label]=dict(error=f'{label}_style.log');continue
                metrics=json.loads(dest.read_text())
                item[label]=dict(pose=metrics['policy'],self_baseline=metrics['pose_window_mmd_reference'])
            summary.append(item)
        temp=trial/'style_comparison.tmp';temp.write_text(json.dumps(summary,indent=2)+'\n');temp.replace(trial/'style_comparison.json')
        status=json.loads((trial/'status.json').read_text())
        if status['status'] in ('complete','failed'):
            return
        time.sleep(15)

if __name__=='__main__':main()
