"""Motion-derived command ablation; expert AMP sampling remains untouched."""
import torch


class MotionCommands:
    def __init__(self, expert, seconds=1.):
        if getattr(expert,'profile',None) != 'physhsi':
            raise ValueError('Motion commands require the native PhysHSI expert library')
        self.device = expert.device
        width = round(seconds * expert.fps)
        if width < 1:
            raise ValueError('Command averaging window must be positive')
        pools, metadata = [], []
        bounds = torch.tensor([.8,.8,.5],device=self.device)
        # AMP columns are the same root-frame velocities used by the task.
        for i,entry in enumerate(expert.entries):
            start,end = int(expert.clip_start[i]),int(expert.clip_end[i])+1
            velocity = expert.amp_frames[start:end][:,[45,46,50]]
            if len(velocity) < width:
                raise ValueError('Motion shorter than command averaging window')
            cumulative = torch.cat((torch.zeros(1,3,device=self.device,dtype=torch.float64),velocity.double().cumsum(0)))
            average = ((cumulative[width:]-cumulative[:-width])/width).float()
            keep = torch.isfinite(average).all(-1) & (average.abs() <= bounds).all(-1)
            pool = average[keep]
            if not len(pool):
                raise ValueError(f'No eligible commands in {entry["path"]}')
            pools.append(pool)
            metadata.append(dict(direction=entry['direction'],candidate_count=len(average),eligible_count=len(pool),
                excluded_fraction=1-len(pool)/len(average),mean=pool.mean(0).tolist(),
                quantiles_05_50_95=torch.quantile(pool,torch.tensor([.05,.5,.95],device=self.device),dim=0).tolist(),
                below_source_linear_deadband_fraction=float((pool[:,:2].norm(dim=-1)<.2).float().mean())))
        self.values = torch.cat(pools)
        self.lengths = torch.tensor([len(p) for p in pools],device=self.device)
        self.starts = torch.cat((torch.zeros(1,dtype=torch.long,device=self.device),self.lengths.cumsum(0)[:-1]))
        self.metadata = dict(profile='motion_1s',seconds=seconds,native_frames=width,
            axes=['vx','vy','wz'],clip_probabilities=[1/len(pools)]*len(pools),clips=metadata,
            deadband=False,velocity_clipping=False,range_filter=[.8,.8,.5],rsi_coupling=False,
            amp_sampling_changed=False,resampling_seconds=8.)

    @torch.no_grad()
    def sample(self, batch, return_motion_ids=False):
        ids = torch.randint(len(self.lengths),(batch,),device=self.device)
        offset = (torch.rand(batch,device=self.device)*self.lengths[ids]).long()
        values = self.values[self.starts[ids]+offset]
        return (values,ids) if return_motion_ids else values
