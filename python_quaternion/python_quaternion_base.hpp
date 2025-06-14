/**
 * @file python_quaternion_base.hpp
 * @brief Quaternion arithmetic and utility functions for C++ with Python-style
 * API.
 *
 * This header defines the `PythonQuaternion` namespace, which provides a
 * template-based Quaternion class and related operations. The implementation is
 * designed to be compatible with Python-style APIs and supports basic
 * quaternion arithmetic, normalization, conjugation, and unit conversion
 * between radians and degrees. The code leverages a dense matrix type for
 * internal storage and provides operator overloads for addition, subtraction,
 * and multiplication.
 */
#ifndef __PYTHON_QUATERNION_BASE_HPP__
#define __PYTHON_QUATERNION_BASE_HPP__

#include "base_math.hpp"
#include "python_numpy.hpp"

namespace PythonQuaternion {

constexpr std::size_t QUATERNION_SIZE = 4;
constexpr std::size_t Q_W = 0;
constexpr std::size_t Q_X = 1;
constexpr std::size_t Q_Y = 2;
constexpr std::size_t Q_Z = 3;

/* Unit conversion */

/**
 * @brief Converts an angle from radians to degrees.
 *
 * @tparam T Numeric type of the input angle (e.g., float, double).
 * @param radian Angle value in radians.
 * @return Angle value converted to degrees.
 */
template <typename T> inline auto radian_to_degree(const T &radian) -> T {
  return radian * static_cast<T>(180) / static_cast<T>(Base::Math::PI);
}

/**
 * @brief Converts an angle from degrees to radians.
 *
 * @tparam T Numeric type of the input angle (e.g., float, double).
 * @param degree Angle value in degrees.
 * @return Angle value converted to radians.
 */
template <typename T> inline auto degree_to_radian(const T &degree) -> T {
  return degree * static_cast<T>(Base::Math::PI) / static_cast<T>(180);
}

/**
 * @brief Converts an angle from degrees to radians, with a minimum division
 * value to avoid division by zero.
 *
 * @tparam T Numeric type of the input angle (e.g., float, double).
 * @param degree Angle value in degrees.
 * @param division_min Minimum value to avoid division by zero.
 * @return Angle value converted to radians.
 */
template <typename T> class Quaternion {
public:
  /* Type */
  using Quaternion_Type = PythonNumpy::DenseMatrix_Type<T, QUATERNION_SIZE, 1>;

public:
  /* Constructor */
  Quaternion()
      : _values(), w(_values.access(Q_W, 0)), x(_values.access(Q_X, 0)),
        y(_values.access(Q_Y, 0)), z(_values.access(Q_Z, 0)) {}

  Quaternion(const T &w, const T &x, const T &y, const T &z)
      : _values(), w(_values.access(Q_W, 0)), x(_values.access(Q_X, 0)),
        y(_values.access(Q_Y, 0)), z(_values.access(Q_Z, 0)) {
    _values(Q_W, 0) = w;
    _values(Q_X, 0) = x;
    _values(Q_Y, 0) = y;
    _values(Q_Z, 0) = z;
  }

  /* Copy Constructor */
  Quaternion(const Quaternion<T> &input)
      : _values(input._values), w(input.w), x(input.x), y(input.y), z(input.z) {
  }

  Quaternion<T> &operator=(const Quaternion<T> &input) {
    if (this != &input) {
      this->_values = input._values;
      this->w = input.w;
      this->x = input.x;
      this->y = input.y;
      this->z = input.z;
    }
    return *this;
  }

  /* Move Constructor */
  Quaternion(Quaternion<T> &&input) noexcept
      : _values(std::move(input._values)), w(input._values.access(Q_W, 0)),
        x(input._values.access(Q_X, 0)), y(input._values.access(Q_Y, 0)),
        z(input._values.access(Q_Z, 0)) {}

  Quaternion<T> &operator=(Quaternion<T> &&input) noexcept {
    if (this != &input) {
      this->_values = std::move(input._values);
      this->w = input._values.access(Q_W, 0);
      this->x = input._values.access(Q_X, 0);
      this->y = input._values.access(Q_Y, 0);
      this->z = input._values.access(Q_Z, 0);
    }
    return *this;
  }

public:
  /* Function */

  /**
   * @brief Creates an identity quaternion.
   *
   * @return An identity quaternion with w=1, x=0, y=0, z=0.
   */
  static inline auto identity() -> Quaternion<T> {
    return Quaternion<T>(1, 0, 0, 0);
  }

  /**
   * @brief Returns the conjugate of the quaternion.
   *
   * The conjugate of a quaternion (w, x, y, z) is defined as (w, -x, -y, -z).
   * This operation negates the vector part (x, y, z) while keeping the scalar
   * part (w) unchanged.
   *
   * @return Quaternion<T> The conjugated quaternion.
   */
  inline auto conjugate() const -> Quaternion<T> {
    return Quaternion<T>(w, -x, -y, -z);
  }

  /**
   * @brief Calculates the norm (magnitude) of the quaternion.
   *
   * This function computes the Euclidean norm of the quaternion, defined as
   * sqrt(w^2 + x^2 + y^2 + z^2), where w, x, y, and z are the quaternion
   * components.
   *
   * @return The norm (magnitude) of the quaternion as a value of type T.
   */
  inline auto norm() const -> T {
    return Base::Math::sqrt(w * w + x * x + y * y + z * z);
  }

  /**
   * @brief Normalizes the quaternion.
   *
   * This function normalizes the quaternion by dividing each component by the
   * norm of the quaternion. If the norm is zero, it returns a quaternion with
   * all components set to zero.
   *
   * @return A normalized quaternion.
   */
  inline auto normalized(const T &division_min) const -> Quaternion<T> {
    T norm_inv = Base::Math::rsqrt(w * w + x * x + y * y + z * z, division_min);
    return Quaternion<T>(w * norm_inv, x * norm_inv, y * norm_inv,
                         z * norm_inv);
  }

private:
  /* Variable */
  Quaternion_Type _values;

public:
  /* Variable */
  T &w;
  T &x;
  T &y;
  T &z;
};

/* Quaternion Addition */

/** * @brief Adds two quaternions.
 *
 * This function performs quaternion addition by adding the corresponding
 * components of the two quaternions.
 *
 * @tparam T Numeric type of the quaternion components (e.g., float, double).
 * @param q1 The first quaternion.
 * @param q2 The second quaternion.
 * @return A new quaternion that is the sum of q1 and q2.
 */
template <typename T>
inline auto operator+(const Quaternion<T> &q1, const Quaternion<T> &q2)
    -> Quaternion<T> {
  return Quaternion<T>(q1.w + q2.w, q1.x + q2.x, q1.y + q2.y, q1.z + q2.z);
}

/* Quaternion Subtraction */

/** * @brief Subtracts two quaternions.
 *
 * This function performs quaternion subtraction by subtracting the
 * corresponding components of the second quaternion from the first.
 *
 * @tparam T Numeric type of the quaternion components (e.g., float, double).
 * @param q1 The first quaternion.
 * @param q2 The second quaternion.
 * @return A new quaternion that is the result of q1 - q2.
 */
template <typename T>
inline auto operator-(const Quaternion<T> &q1, const Quaternion<T> &q2)
    -> Quaternion<T> {
  return Quaternion<T>(q1.w - q2.w, q1.x - q2.x, q1.y - q2.y, q1.z - q2.z);
}

/* Quaternion Product */

/** * @brief Multiplies two quaternions.
 *
 * This function performs quaternion multiplication using the Hamilton product
 * formula. The result is a new quaternion that represents the combined rotation
 * of the two input quaternions.
 *
 * @tparam T Numeric type of the quaternion components (e.g., float, double).
 * @param q1 The first quaternion.
 * @param q2 The second quaternion.
 * @return A new quaternion that is the product of q1 and q2.
 */
template <typename T>
inline auto operator*(const Quaternion<T> &q1, const Quaternion<T> &q2)
    -> Quaternion<T> {

  Quaternion<T> result;

  result.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z;
  result.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y;
  result.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x;
  result.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w;

  return result;
}

/* Quaternion divide */

/**
 * @brief Divides one quaternion by another, with protection against division by
 * zero.
 *
 * This function computes the division of quaternion q1 by quaternion q2. It
 * does so by multiplying q1 by the conjugate of q2, scaled by the inverse
 * squared norm of q2. To avoid division by zero or very small values, the
 * denominator is clamped to a minimum value specified by division_min.
 *
 * @tparam T The numeric type used for the quaternion components (e.g., float,
 * double).
 * @param q1 The dividend quaternion.
 * @param q2 The divisor quaternion.
 * @param division_min The minimum value allowed for the denominator to prevent
 * division by zero.
 * @return Quaternion<T> The result of the division q1 / q2.
 */
template <typename T>
inline auto quaternion_divide(const Quaternion<T> &q1, const Quaternion<T> &q2,
                              const T &division_min) -> Quaternion<T> {

  auto q2_conj = q2.conjugate();

  T norm_inv_square = Base::Utility::avoid_zero_divide(
      static_cast<T>(1) / (q2_conj.w * q2_conj.w + q2_conj.x * q2_conj.x +
                           q2_conj.y * q2_conj.y + q2_conj.z * q2_conj.z),
      division_min);

  q2_conj.w *= norm_inv_square;
  q2_conj.x *= norm_inv_square;
  q2_conj.y *= norm_inv_square;
  q2_conj.z *= norm_inv_square;

  return q1 * q2_conj;
}

/* Quaternion Type */
template <typename T> using Quaternion_Type = Quaternion<T>;

/* make Quaternion */

/**
 * @brief Creates a quaternion from its components.
 *
 * This function constructs a quaternion using the provided components w, x, y,
 * and z.
 *
 * @tparam T Numeric type of the quaternion components (e.g., float, double).
 * @param w The scalar part of the quaternion.
 * @param x The first component of the vector part.
 * @param y The second component of the vector part.
 * @param z The third component of the vector part.
 * @return A Quaternion<T> object initialized with the given components.
 */
template <typename T>
inline auto make_quaternion(const T &w, const T &x, const T &y, const T &z)
    -> Quaternion<T> {
  return Quaternion<T>(w, x, y, z);
}

} // namespace PythonQuaternion

#endif // __PYTHON_QUATERNION_BASE_HPP__
