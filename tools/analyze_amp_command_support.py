"""Offline audit of source command support and independent pose descriptors.
No simulator rollout or training is launched. Coverage thresholds are diagnostics.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from blinkamp.physhsi_motion import PhysHSIMotionLibrary
from blinkmimic.kinematics import Kinematics
from blinkmimic.robot import get_robot
from blinkmimic.mdp import unflatten_state
from blinkamp.amp import MotionDiscriminator


def summary(x):
    return dict(mean=x.mean(0).tolist(),quantiles=torch.quantile(x,torch.tensor([.05,.5,.95]),dim=0).tolist())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--record-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(4);torch.manual_seed(2026);np.random.seed(2026)
    robot=get_robot('g1');kin=Kinematics(robot,'cpu');lib=PhysHSIMotionLibrary(kin,robot)
    report=dict(velocity_axes=['vx','vy','wz'],clips={},coverage={},policy={})
    for i,entry in enumerate(lib.entries):
        frames=lib.amp_frames[lib.clip_start[i]:lib.clip_end[i]+1]
        q=lib.q[lib.clip_start[i]:lib.clip_end[i]+1]
        mask=(q<kin.q_lo)|(q>kin.q_hi)
        fraction=mask.float().mean(0)
        violation=(torch.relu(kin.q_lo-q)+torch.relu(q-kin.q_hi)).amax(-1)
        report['clips'][entry['direction']]=dict(frame_velocity=summary(frames[:,[45,46,50]]),
            outside_hard_limit_frame_fraction=float(mask.any(-1).float().mean()),
            hard_violation_rad=summary(violation[:,None]),
            hard_violation_over_005_fraction=float((violation>.05).float().mean()),
            outside_hard_limits={robot.joint_names[j]:float(fraction[j]) for j in range(29) if fraction[j]>0})
    windows=lib.sample_windows(4096)
    support=windows[:,:,[45,46,50]].mean(1)
    commands=torch.empty(50000,3);commands[:,:2].uniform_(-.8,.8);commands[:,2].uniform_(-.5,.5)
    commands[:,:2]*=(commands[:,:2].norm(dim=-1)>.2)[:,None];commands[:,2]*=commands[:,2].abs()>.2
    report['commands']=summary(commands)
    report['expert_window_velocity']=summary(support)
    for planar,yaw in [(.1,.1),(.2,.2),(.3,.3)]:
        inside=[]
        for chunk in commands.split(512):
            distance=torch.cdist(chunk[:,:2],support[:,:2])
            angular=(chunk[:,None,2]-support[None,:,2]).abs()
            inside.append(((distance<=planar)&(angular<=yaw)).any(1))
        covered=torch.cat(inside)
        report['coverage'][f'planar_{planar}_yaw_{yaw}']=dict(command_fraction=float(covered.float().mean()),
            positive_y_fraction=float(covered[commands[:,1]>.2].float().mean()),
            negative_y_fraction=float(covered[commands[:,1]<-.2].float().mean()))
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=ck['config']
    disc=MotionDiscriminator(normalize=cfg['disc_normalize'],activation=cfg['disc_activation'],reward_mode=cfg['amp_reward'])
    disc.load_state_dict(ck['models']['disc']);disc.eval().requires_grad_(False)
    pose_columns=list(range(45))+list(range(51,57))
    mean=lib.amp_frames[:,pose_columns].mean(0);std=lib.amp_frames[:,pose_columns].std(0).clamp(min=.05)
    descriptors=lambda f:dict(height=float(f[:,0].mean()),hand_forward=float(f[:,[30,33]].mean()),
        tilt_tangent_z=float(f[:,53].mean()),knee=float(f[:,[4,10]].mean()))
    report['expert_pose']=descriptors(lib.amp_frames)
    with torch.no_grad():
        report['expert_discriminator']=dict(score=float(disc(windows).mean()),reward=float(disc.reward(windows).mean()))
        for j in range(7):
            path=args.record_root/f'model_command_{j:02d}.npz'
            if not path.exists():continue
            raw=np.load(path);S=unflatten_state(torch.tensor(raw['states'][100:]));f=lib.features.amp(S)
            w=f.unfold(0,10,1).transpose(-1,-2)
            selected=w[torch.linspace(0,len(w)-1,min(256,len(w))).long()]
            pose=((selected[:,:,pose_columns]-mean)/std).flatten(1)
            expert_pose=((windows[:512,:,pose_columns]-mean)/std).flatten(1)
            nearest=torch.cdist(pose,expert_pose).amin(1)/(pose.shape[1]**.5)
            # Descriptive intervention only: this changes feature blocks, not a physical trajectory.
            velocity_only=selected.clone();velocity_only[:,:,45:51]=windows[:len(selected),:,45:51]
            report['policy'][str(j)]=dict(command=raw['commands'][0].tolist(),pose=descriptors(f),
                score=float(disc(selected).mean()),reward=float(disc.reward(selected).mean()),
                score_with_expert_velocity_block=float(disc(velocity_only).mean()),
                expert_normalized_pose_nearest_rms=float(nearest.mean()))
    report['limitations']=['Coverage uses 4096 sampled expert windows and heuristic tolerances; not an exact support boundary.',
        'Policy descriptors use one recorded replica per command after the first 2 seconds.',
        'Feature velocity replacement is an unphysical sensitivity probe, not a causal training result.',
        'Pose nearest distance includes no root velocities but is not a held-out perceptual style metric.']
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
