import os
import sys
sys.path.append(os.getcwd())

import math
import quaternion
import numpy as np


def identity():
    return np.quaternion(1.0, 0.0, 0.0, 0.0)


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
