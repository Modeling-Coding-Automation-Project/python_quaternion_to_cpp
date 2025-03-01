#include <iostream>

#include "python_quaternion.hpp"

using namespace PythonQuaternion;

int main(void) {
  /* Define Quaternion */
  auto q_a = make_quaternion(1.0, 2.0, 3.0, 4.0);

  // Identity
  auto q_b = Quaternion<double>::identity();

  // Substitution
  q_b.w = 5.0;
  q_b.x = 6.0;
  q_b.y = 7.0;
  q_b.z = 8.0;

  // Quaternion product
  auto q_mul = q_a * q_b;

  std::cout << "Quaternion product: ";
  std::cout << q_mul.w << ", ";
  std::cout << q_mul.x << ", ";
  std::cout << q_mul.y << ", ";
  std::cout << q_mul.z << std::endl << std::endl;

  return 0;
}
