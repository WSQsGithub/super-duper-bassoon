"""Read real actuator outputs and reconstruct the exact pre-force-limit MuJoCo signal."""

import mujoco
import numpy as np
import torch

from blinkmimic.g1.g1_constants import motor_of


class ActuationLabels:
    def __init__(self, env):
        self.env = env
        self.ids = env.robot.indexing.ctrl_ids
        ids = self.ids.cpu().numpy()
        m = env.sim.mj_model
        expected = [m.joint(f'robot/{name}').id for name in env.spec.joint_names]
        if not np.array_equal(m.actuator_trnid[ids, 0], expected):
            raise ValueError('Actuator/joint order mismatch')
        if not (np.all(m.actuator_trntype[ids] == mujoco.mjtTrn.mjTRN_JOINT)
                and np.allclose(m.actuator_gear[ids, 0], 1)
                and np.allclose(m.actuator_gear[ids, 1:], 0)
                and np.all(m.actuator_gaintype[ids] == mujoco.mjtGain.mjGAIN_FIXED)
                and np.all(m.actuator_biastype[ids] == mujoco.mjtBias.mjBIAS_AFFINE)
                and np.all(m.actuator_dyntype[ids] == mujoco.mjtDyn.mjDYN_NONE)):
            raise ValueError('Torque labels require the validated direct-drive built-in PD actuator type')
        if env.spec.name != 'g1':
            raise ValueError('Velocity limits are currently specified for G1 only')
        dev = env.device
        self.kp = torch.tensor(m.actuator_gainprm[ids, 0].copy(), dtype=torch.float32, device=dev)
        self.torque_limits = torch.tensor(np.abs(m.actuator_forcerange[ids]).max(-1).copy(), dtype=torch.float32, device=dev)
        self.velocity_limits = torch.tensor([motor_of(name)[1] for name in env.spec.joint_names], device=dev)
        self.ctrl_limited = torch.tensor(m.actuator_ctrllimited[ids].copy(), device=dev)

    def read(self):
        e, ids = self.env, self.ids
        d, m = e.sim.data, e.sim.model
        gain, bias = m.actuator_gainprm[:, ids], m.actuator_biasprm[:, ids]
        ctrl, ranges = d.ctrl[:, ids], m.actuator_ctrlrange[:, ids]
        ctrl = torch.where(self.ctrl_limited, torch.clamp(ctrl, ranges[..., 0], ranges[..., 1]), ctrl)
        # These are the same last-substep lengths/velocities used in Warp's
        # _actuator_force kernel, not an approximation using the 50Hz next state.
        demand = (gain[..., 0] * ctrl + bias[..., 0] + bias[..., 1] * d.actuator_length[:, ids]
                  + bias[..., 2] * d.actuator_velocity[:, ids])
        actual = e.robot.data.actuator_force
        return actual.clone(), demand.clone()
