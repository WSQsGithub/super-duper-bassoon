import csv
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
import torch

ROOT = Path(__file__).resolve().parents[1]
def module(name):
    spec = importlib.util.spec_from_file_location(name,ROOT/'tools'/f'{name}.py')
    out = importlib.util.module_from_spec(spec);spec.loader.exec_module(out);return out
watch = module('watch_amp_training')
runner = module('run_amp_suite')

class WatchdogContract(unittest.TestCase):
    def test_tracking_and_curve_stability_both_required(self):
        report = dict(iteration=200000,results=[dict(survival_rate=1.,velocity_rmse_m_s=.1,yaw_rmse_rad_s=.1)])
        rows = [dict(iteration=str(i),velocity_error='.1',yaw_error='.1',real_reward='.1')
                for i in range(160050,200001,50)]
        self.assertEqual(watch.continuation_reasons(report,rows),[])
        report['results'][0]['velocity_rmse_m_s']=.3
        self.assertIn('velocity_rmse_m_s_above_0.2',watch.continuation_reasons(report,rows))
        report['results'][0]['velocity_rmse_m_s']=.1
        rows[-1]['real_reward']='2'
        self.assertIn('real_reward_still_changing',watch.continuation_reasons(report,rows))
        self.assertEqual(watch.continuation_reasons({},rows),['missing_evaluation'])

    def test_bad_newest_checkpoint_falls_back_without_deleting_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            for i,value in [(5000,1.),(10000,float('nan'))]:
                torch.save(dict(models={'x':torch.tensor(value)},optimizers={},nonfinite=1,iteration=i),p/f'ckpt_{i:06d}.pt')
            path,ck,rejected = runner.usable_checkpoint(p.glob('*.pt'),1)
            self.assertEqual(ck['iteration'],5000)
            self.assertEqual(len(rejected),1)
            self.assertTrue((p/'ckpt_010000.pt').exists())
            with self.assertRaises(RuntimeError):
                runner.usable_checkpoint(p.glob('*.pt'),0)

    def test_csv_recovery_preserves_future_logs_separately(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'train_log.csv'
            p.write_text('iteration,nonfinite\n5000,1\n5050,1\n')
            runner.trim_future_log(p,5000)
            with p.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r['iteration'] for r in rows],['5000'])
            self.assertEqual(len(list(Path(d).glob('train_log.before_resume_*.csv'))),1)

    def test_process_identity_check(self):
        self.assertTrue(watch.alive(os.getpid(),'python'))
        self.assertFalse(watch.alive(os.getpid(),'not-the-training-process'))
        self.assertFalse(watch.alive(None,'python'))

if __name__ == '__main__':
    unittest.main()
