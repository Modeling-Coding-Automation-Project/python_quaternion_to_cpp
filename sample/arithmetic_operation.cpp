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

  // Quaternion divide
  auto q_div = quaternion_divide(q_a, q_b, 1.0e-10);

  std::cout << "Quaternion divide: ";
  std::cout << q_div.w << ", ";
  std::cout << q_div.x << ", ";
  std::cout << q_div.y << ", ";
  std::cout << q_div.z << std::endl << std::endl;

  // Conjugate
  auto q_conj = q_a.conjugate();

  std::cout << "Conjugate: ";
  std::cout << q_conj.w << ", ";
  std::cout << q_conj.x << ", ";
  std::cout << q_conj.y << ", ";
  std::cout << q_conj.z << std::endl << std::endl;

  return 0;
}
