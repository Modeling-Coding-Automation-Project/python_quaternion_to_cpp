"""
This module provides functions for quaternion-based rotation calculations, specifically for integrating 3-axis angular velocity (gyroscope data) to update orientation using quaternions. The main function, `integrate_gyro`, computes the new orientation quaternion by integrating angular velocity over a time step, ensuring strict normalization and numerical stability.
"""
import os
import sys
sys.path.append(os.getcwd())

import math
import quaternion
import numpy as np


DEFAULT_DIVISION_MIN = 1.0e-10


def identity():
    '''
    # Returns the identity quaternion
    Details:
        The identity quaternion is a quaternion representing no rotation.
        It is defined as q.w = 1, q.x = 0, q.y = 0, q.z = 0.
    '''
    return np.quaternion(1.0, 0.0, 0.0, 0.0)


def q_from_rotation_vector(theta, direction_vector):
    '''
    # Returns a quaternion from a rotation vector
    Arguments:
        theta: Rotation angle in radians
        direction_vector: Direction vector of the rotation axis
    Returns:
        Quaternion representing the rotation
    Details:
        Converts a rotation vector defined by an angle and a direction vector
        into a quaternion. The direction vector should be normalized.
    '''

    direction_vector_sq = direction_vector[0] * direction_vector[0] + \
        direction_vector[1] * direction_vector[1] + \
        direction_vector[2] * direction_vector[2]

    direction_vector_norm_inv = 0
    if direction_vector_sq < DEFAULT_DIVISION_MIN:
        direction_vector_norm_inv = 1.0 / math.sqrt(DEFAULT_DIVISION_MIN)
    else:
        direction_vector_norm_inv = 1.0 / math.sqrt(direction_vector_sq)

    half_theta = 0.5 * theta
    sin_half_theta = math.sin(half_theta)
    norm_inv_sin_theta = direction_vector_norm_inv * sin_half_theta

    w = math.cos(half_theta)
    x = direction_vector[0] * norm_inv_sin_theta
    y = direction_vector[1] * norm_inv_sin_theta
    z = direction_vector[2] * norm_inv_sin_theta

    return np.quaternion(w, x, y, z)


def rotation_matrix_euler_absolute(roll, pitch, yaw):
    '''
    # Get rotation matrix from Euler angles

    Note:
        The rotation order is ZYX.
        The rotation is done with the axis fixed to the ground.
    '''

    R = np.zeros((3, 3))

    roll_sin = math.sin(roll)
    roll_cos = math.cos(roll)
    pitch_sin = math.sin(pitch)
    pitch_cos = math.cos(pitch)
    yaw_sin = math.sin(yaw)
    yaw_cos = math.cos(yaw)

    R[0, 0] = yaw_cos * pitch_cos
    R[0, 1] = -yaw_sin * pitch_cos
    R[0, 2] = pitch_sin
    R[1, 0] = yaw_cos * pitch_sin * roll_sin + yaw_sin * roll_cos
    R[1, 1] = -yaw_sin * pitch_sin * roll_sin + yaw_cos * roll_cos
    R[1, 2] = -pitch_cos * roll_sin
    R[2, 0] = -yaw_cos * pitch_sin * roll_cos + yaw_sin * roll_sin
    R[2, 1] = yaw_sin * pitch_sin * roll_cos + yaw_cos * roll_sin
    R[2, 2] = pitch_cos * roll_cos

    return R


def integrate_gyro_approximately(omega, q, time_step):
    '''
    # Integrates 3-axis angular velocity to obtain a quaternion (approximate)

    Argument:
        w: 3-axis angular velocity [rad/s]
        q: Quaternion one step before
        time_step: Integration time step [s]

        Returns:
            q: Quaternion

    Details:
        Integrates 3-axis angular velocity using an approximate
        formula to obtain a quaternion.
        The angular velocity is the angular velocity around each axis of XYZ, and
        XYZ axis is assumed to be right-handed.
        The quaternion is a quaternion representing the posture angle.
        q.w ^ 2 + q.x ^ 2 + q.y ^ 2 + q.z ^ 2 = 1 must be satisfied.
        The initial posture is q.w = 1, q.x = 0, q.y = 0, q.z = 0 by default.
    '''

    out_q = identity()
    omega_x = omega[0, 0]
    omega_y = omega[1, 0]
    omega_z = omega[2, 0]

    out_q.w = (-(omega_x * q.x) - omega_y * q.y -
               omega_z * q.z) * 0.5 * time_step + q.w
    out_q.x = (omega_x * q.w + omega_z * q.y -
               omega_y * q.z) * 0.5 * time_step + q.x
    out_q.y = (omega_y * q.w - omega_z * q.x +
               omega_x * q.z) * 0.5 * time_step + q.y
    out_q.z = (omega_z * q.w + omega_y * q.x -
               omega_x * q.y) * 0.5 * time_step + q.z

    return out_q.normalized()


def integrate_gyro(omega, q, time_step, division_min):
    '''
    # Integrates 3-axis angular velocity to obtain a quaternion (strict)

    Arguments:
        omega: 3-axis angular velocity [rad/s]
        q: Quaternion one step before
        time_step: Integration time step [s]

    Returns:
        q: Quaternion

    Details:
        Integrates 3-axis angular velocity using a strict formula to obtain a quaternion.
        The angular velocity is the angular velocity around each axis of XYZ, and
        XYZ axis is assumed to be right-handed.
        The quaternion is a quaternion representing the posture angle.
        q.w ^ 2 + q.x ^ 2 + q.y ^ 2 + q.z ^ 2 = 1 must be satisfied.
        The initial posture is q.w = 1, q.x = 0, q.y = 0, q.z = 0 by default.
    '''
    X = 0
    Y = 1
    Z = 2

    w_norm_inv = (omega[X, 0] * omega[X, 0] + omega[Y, 0]
                  * omega[Y, 0] + omega[Z, 0] * omega[Z, 0])

    if w_norm_inv < division_min:
        w_norm_inv = 1.0 / math.sqrt(division_min)
    else:
        w_norm_inv = 1.0 / math.sqrt(w_norm_inv)

    w_sin = math.sin(0.5 * time_step / w_norm_inv)
    w_cos = math.cos(0.5 * time_step / w_norm_inv)

    qw = np.quaternion(w_cos, omega[X, 0] * w_norm_inv * w_sin,
                       omega[Y, 0] * w_norm_inv * w_sin, omega[Z, 0] * w_norm_inv * w_sin)

    out_q = q * qw

    return out_q.normalized()
