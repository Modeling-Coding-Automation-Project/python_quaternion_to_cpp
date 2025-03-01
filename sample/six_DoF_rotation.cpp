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

  return 0;
}
