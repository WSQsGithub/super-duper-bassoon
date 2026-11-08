from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from blinkmimic.robot import get_robot
from tools.audit_amp_mapping import compare_fk, read_mapping
from tools.import_amp_data import parse_joint_mapping


class IndependentMappingContract(unittest.TestCase):
    def test_numeric_column_ids_are_not_file_line_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'map.txt'
            path.write_text('2 knee\n0 hip_pitch\n1 hip_roll\n')
            self.assertEqual(read_mapping(path), ['hip_pitch', 'hip_roll', 'knee'])
            self.assertEqual(parse_joint_mapping(path), ['hip_pitch', 'hip_roll', 'knee'])
            path.write_text('0 hip\n0 knee\n')
            with self.assertRaises(ValueError):
                read_mapping(path)
            with self.assertRaises(ValueError):
                parse_joint_mapping(path)

    def test_independent_urdf_fk_matches_and_detects_a_wrong_permutation(self):
        robot = get_robot('g1')
        source = ET.parse('data/blinkamp/reference/g1_29dof.urdf').getroot()
        model = mujoco.MjModel.from_xml_path(robot.mjcf_path)
        mapping = read_mapping('data/blinkamp/joint_id.txt')
        q0 = np.array(robot.default_joint_pos_list())
        probes = [q0] + [q0 + .2 * np.eye(29)[j] for j in range(29)]
        correct = compare_fk(source, model, mapping, list(robot.joint_names), probes)
        self.assertLess(correct['max_body_position_error_m'], 1e-5)
        self.assertLess(correct['max_rotation_matrix_error'], 1e-5)
        swapped = list(robot.joint_names)
        swapped[1], swapped[3] = swapped[3], swapped[1]
        wrong = compare_fk(source, model, mapping, swapped, probes)
        self.assertGreater(wrong['max_body_position_error_m'], .1)


if __name__ == '__main__':
    unittest.main()
