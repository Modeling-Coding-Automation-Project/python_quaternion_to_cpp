"""
File: six_DoF_rotation.py

This script demonstrates various 6 Degrees of Freedom (DoF) rotation operations using both Euler angles and quaternions. It utilizes the numpy and quaternion libraries, along with a custom PythonRotation module, to perform and compare different rotation representations and integrations.
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import quaternion
# Below is the module in "python_quaternion" directory.
import python_quaternion.rotation as PythonRotation

# %% Euler angles rotation
R_euler = PythonRotation.rotation_matrix_euler_absolute(
    0.0, np.pi / 2, 0.0)
print("R_euler:\n", R_euler)

v = np.array([[1.0], [0.0], [0.0]])
v_rotated = R_euler @ v
print("v_rotated:\n", v_rotated)

# %% quaternion rotation successive
q_1 = PythonRotation.q_from_rotation_vector(
    0.0, np.array([[1.0], [0.0], [0.0]]))

q_2 = PythonRotation.q_from_rotation_vector(
    np.pi / 2, np.array([[0.0], [1.0], [0.0]]))

q_3 = PythonRotation.q_from_rotation_vector(
    np.pi / 2, np.array([[0.0], [0.0], [1.0]]))

# q_123 = q_3 * q_2 * q_1
q_123 = q_1 * q_2 * q_3

print("q_123:", q_123)
R_q_123 = quaternion.as_rotation_matrix(q_123)
v_rotated = R_q_123 @ v
print("v_rotated:\n", v_rotated)

# %% quaternion rotation integration
q_1 = PythonRotation.identity()
omega_1 = np.array([[1.0], [-1.0], [1.0]])

q_1_next_approximate = PythonRotation.integrate_gyro_approximately(
    omega_1, q_1, 0.1)
print("q_1_next_approximate:", q_1_next_approximate)

q_1_next = PythonRotation.integrate_gyro(omega_1, q_1, 0.1, 1.0e-10)
print("q_1_next:", q_1_next)

# %% Rotation matrix
R_q_1_next = quaternion.as_rotation_matrix(q_1_next)
print("R_q_1_next:\n", R_q_1_next)

# %% Quaternion from euler angles
# This is not the same as "as_euler_angles()" in "python_quaternion_rotation.hpp".
# The order of the Euler angles is different.
euler_angles_q_1 = quaternion.as_euler_angles(q_1_next)
print("euler_angles:", euler_angles_q_1)
