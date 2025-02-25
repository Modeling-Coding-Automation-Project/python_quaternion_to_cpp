import numpy as np
import quaternion
import math

q1 = np.quaternion(1.0, 2.0, 3.0, 4.0)
q2 = np.quaternion(5.0, 6.0, 7.0, 8.0)

print(q1 * q2)

print("q1.w: ", q1.w)
print("q1.x: ", q1.x)
print("q1.y: ", q1.y)
print("q1.z: ", q1.z)

q1_conj = q1.conjugate()
print("q1_conj: ", q1_conj)

q1_norm = math.sqrt(q1.norm())  # norm of numpy-quaternion is squared norm.
print("q1_norm: ", q1_norm)

q1_normalized = q1.normalized()
print("q1_normalized: ", q1_normalized)

q_divide = q1 / q2
print("q_divide: ", q_divide)
