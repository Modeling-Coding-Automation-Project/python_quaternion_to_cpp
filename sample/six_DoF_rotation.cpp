#include <iostream>

#include "python_quaternion.hpp"

using namespace PythonQuaternion;

int main(void) {
  /* Integrate angular velocity */
  auto q_now = make_quaternion(1.0, 0.0, 0.0, 0.0);
  auto omega = make_Omega(1.0, -1.0, 1.0);
  double dt = 0.1;

  auto q_next = integrate_gyro(omega, q_now, dt, 1.0e-10);

  std::cout << "Integrate angular velocity: ";
  std::cout << q_next.w << ", ";
  std::cout << q_next.x << ", ";
  std::cout << q_next.y << ", ";
  std::cout << q_next.z << std::endl << std::endl;

  /* Integrate angular velocity approximate */
  auto q_next_a = integrate_gyro_approximately(omega, q_now, dt, 1.0e-10);

  std::cout << "Integrate angular velocity approximate: ";
  std::cout << q_next_a.w << ", ";
  std::cout << q_next_a.x << ", ";
  std::cout << q_next_a.y << ", ";
  std::cout << q_next_a.z << std::endl << std::endl;

  /* Rotation Matrix from Quaternion */
  auto R_q_1_next = as_rotation_matrix(q_next);

  std::cout << "Rotation Matrix from Quaternion: " << std::endl;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      std::cout << R_q_1_next(i, j) << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  /* Euler angle from Quaternion */
  auto euler_angle = as_euler_angles(q_next);

  std::cout << "Euler angle from Quaternion: ";
  std::cout << euler_angle(0, 0) << ", ";
  std::cout << euler_angle(1, 0) << ", ";
  std::cout << euler_angle(2, 0) << std::endl << std::endl;

  return 0;
}
