"""Render a BlinkAMP --record trajectory with the local G1 model and EGL."""

import argparse
import os
from pathlib import Path
import re

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
import imageio.v2 as imageio

from blinkmimic.robot import get_robot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trajectory', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--fps', type=int, default=25)
    args = parser.parse_args()
    saved = np.load(args.trajectory)
    states = saved['states']
    robot = get_robot('g1')
    path = Path(robot.mjcf_path)
    xml = path.read_text()
    xml = re.sub(r'meshdir="([^"]*)"', lambda m: f'meshdir="{path.parent / m.group(1)}"', xml)
    xml = xml.replace('</worldbody>', '<light pos="0 0 5" dir="0 0 -1"/>'
                      '<geom type="plane" size="0 0 .05" rgba=".8 .85 .9 1"/></worldbody>')
    xml = xml.replace('<asset>', '<visual><global offwidth="960" offheight="540"/></visual><asset>', 1)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 2.8, 135, -15
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with mujoco.Renderer(model, height=540, width=960) as renderer, imageio.get_writer(args.out, fps=args.fps) as writer:
        indices = np.arange(0, len(states) * float(saved['dt']), 1 / args.fps) / float(saved['dt'])
        for i in indices.astype(int).clip(max=len(states) - 1):
            S = states[i]
            data.qpos[:7] = S[:7]
            for j, name in enumerate(robot.joint_names):
                data.qpos[model.jnt_qposadr[model.joint(name).id]] = S[13 + j]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [S[0], S[1], .65]
            renderer.update_scene(data, camera)
            writer.append_data(renderer.render())
    print(args.out)


if __name__ == '__main__':
    main()
