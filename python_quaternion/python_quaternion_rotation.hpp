/**
 * @file python_quaternion_rotation.hpp
 * @brief Quaternion and rotation utilities for 3D vector and orientation
 * operations.
 *
 * This header provides a set of classes and functions for representing and
 * manipulating 3D vectors, quaternions, and rotation matrices, with a focus on
 * integration with Python-like data structures and conventions. The utilities
 * include conversions between rotation representations, integration of angular
 * velocity to update orientation, and construction of common rotation-related
 * types.
 */
#ifndef __PYTHON_QUATERNION_ROTATION_HPP__
#define __PYTHON_QUATERNION_ROTATION_HPP__

#include "python_numpy.hpp"
#include "python_quaternion_base.hpp"

namespace PythonQuaternion {

constexpr std::size_t CARTESIAN_SIZE = 3;
constexpr std::size_t P_X = 0;
constexpr std::size_t P_Y = 1;
constexpr std::size_t P_Z = 2;

/* Cartesian Vector */

/**
 * @brief A 3D Cartesian vector class template.
 *
 * This class represents a 3-dimensional vector with x, y, and z components,
 * inheriting from PythonNumpy::Matrix for dense storage. It provides
 * constructors for default, value-initialized, copy, and move semantics,
 * as well as assignment operators. The x, y, and z members are references
 * to the underlying matrix storage for convenient access.
 *
 * @tparam T The type of the vector components (e.g., float, double).
 *
 * @note
 * - The class assumes the existence of constants P_X, P_Y, P_Z, and
 * CARTESIAN_SIZE.
 * - The base class PythonNumpy::Matrix must provide access and set methods.
 */
template <typename T>
class CartesianVector
    : public PythonNumpy::Matrix<PythonNumpy::DefDense, T, CARTESIAN_SIZE, 1> {
public:
  /* Constructor */
  CartesianVector()
      : PythonNumpy::Matrix<PythonNumpy::DefDense, T, CARTESIAN_SIZE, 1>(),
        x(this->access(P_X, 0)), y(this->access(P_Y, 0)),
        z(this->access(P_Z, 0)) {}

  CartesianVector(const T &x, const T &y, const T &z)
      : PythonNumpy::Matrix<PythonNumpy::DefDense, T, CARTESIAN_SIZE, 1>(),
        x(this->access(P_X, 0)), y(this->access(P_Y, 0)),
        z(this->access(P_Z, 0)) {

    this->template set<P_X, 0>(x);
    this->template set<P_Y, 0>(y);
    this->template set<P_Z, 0>(z);
  }

  /* Copy Constructor */
  CartesianVector(const CartesianVector<T> &input)
      : PythonNumpy::Matrix<PythonNumpy::DefDense, T, CARTESIAN_SIZE, 1>(input),
        x(input.x), y(input.y), z(input.z) {}

  CartesianVector<T> &operator=(const CartesianVector<T> &input) {
    if (this != &input) {
      this->matrix = input.matrix;
      this->x = input.x;
      this->y = input.y;
      this->z = input.z;
    }
    return *this;
  }

  /* Move Constructor */
  CartesianVector(CartesianVector<T> &&input) noexcept
      : PythonNumpy::Matrix<PythonNumpy::DefDense, T, CARTESIAN_SIZE, 1>(
            std::move(input)),
        x(input.access(P_X, 0)), y(input.access(P_Y, 0)),
        z(input.access(P_Z, 0)) {}

  CartesianVector<T> &operator=(CartesianVector<T> &&input) noexcept {
    if (this != &input) {
      this->matrix = std::move(input.matrix);
      this->x = input.access(P_X, 0);
      this->y = input.access(P_Y, 0);
      this->z = input.access(P_Z, 0);
    }
    return *this;
  }

public:
  /* Variable */
  T &x;
  T &y;
  T &z;
};

/* Direction Vector Type */
template <typename T> using DirectionVector_Type = CartesianVector<T>;

/* make Direction Vector */

/**
 * @brief Creates a DirectionVector_Type from x, y, and z components.
 *
 * This function constructs a DirectionVector_Type object using the provided
 * x, y, and z values. It is a convenience function for creating direction
 * vectors in 3D space.
 *
 * @tparam T The type of the vector components (e.g., float, double).
 * @param x The x component of the direction vector.
 * @param y The y component of the direction vector.
 * @param z The z component of the direction vector.
 * @return A DirectionVector_Type object initialized with the given components.
 */
template <typename T>
inline auto make_DirectionVector(const T &x, const T &y, const T &z)
    -> DirectionVector_Type<T> {

  return CartesianVector<T>(x, y, z);
}

/* Quaternion from rotation angle and direction vector */

/**
 * @brief Creates a quaternion from a rotation vector and a direction vector.
 *
 * This function constructs a quaternion representing a rotation defined by
 * an angle (theta) and a direction vector. The direction vector is normalized
 * before being used to compute the quaternion.
 *
 * @tparam T The type of the angle and direction vector components (e.g., float,
 * double).
 * @param theta The rotation angle in radians.
 * @param direction_vector The direction vector defining the axis of rotation.
 * @param division_min The minimum value to avoid division by zero.
 * @return A Quaternion_Type object representing the rotation.
 */
template <typename T>
inline auto
q_from_rotation_vector(const T &theta,
                       const DirectionVector_Type<T> &direction_vector,
                       const T &division_min) -> Quaternion_Type<T> {

  T direction_vector_sq = direction_vector.x * direction_vector.x +
                          direction_vector.y * direction_vector.y +
                          direction_vector.z * direction_vector.z;

  T direction_vector_norm_inv = static_cast<T>(0);

  direction_vector_norm_inv =
      Base::Math::rsqrt(direction_vector_sq, division_min);

  auto half_theta = theta * static_cast<T>(0.5);
  T sin_half_theta = static_cast<T>(0);
  T cos_half_theta = static_cast<T>(0);
  Base::Math::sincos(half_theta, sin_half_theta, cos_half_theta);

  T norm_inv_sin_theta = direction_vector_norm_inv * sin_half_theta;

  return Quaternion_Type<T>(cos_half_theta,
                            direction_vector.x * norm_inv_sin_theta,
                            direction_vector.y * norm_inv_sin_theta,
                            direction_vector.z * norm_inv_sin_theta);
}

/* Omega Type */
template <typename T> using Omega_Type = CartesianVector<T>;

/* make Omega */

/**
 * @brief Creates an Omega_Type from x, y, and z components.
 *
 * This function constructs an Omega_Type object using the provided x, y,
 * and z values. It is a convenience function for creating angular velocity
 * vectors in 3D space.
 *
 * @tparam T The type of the vector components (e.g., float, double).
 * @param x The x component of the angular velocity vector.
 * @param y The y component of the angular velocity vector.
 * @param z The z component of the angular velocity vector.
 * @return An Omega_Type object initialized with the given components.
 */
template <typename T>
inline auto make_Omega(const T &x, const T &y, const T &z) -> Omega_Type<T> {

  return CartesianVector<T>(x, y, z);
}

/* Euler Angle Type */
template <typename T> using EulerAngle_Type = CartesianVector<T>;

/* make Euler Angle */

/**
 * @brief Creates an EulerAngle_Type from roll, pitch, and yaw angles.
 *
 * This function constructs an EulerAngle_Type object using the provided
 * roll, pitch, and yaw angles. It is a convenience function for creating
 * Euler angles in 3D space.
 *
 * @tparam T The type of the angle components (e.g., float, double).
 * @param roll The roll angle in radians.
 * @param pitch The pitch angle in radians.
 * @param yaw The yaw angle in radians.
 * @return An EulerAngle_Type object initialized with the given angles.
 */
template <typename T>
inline auto make_EulerAngle(const T &roll, const T &pitch, const T &yaw)
    -> EulerAngle_Type<T> {

  return CartesianVector<T>(roll, pitch, yaw);
}

/* Rotation Matrix Type */
template <typename T>
using RotationMatrix_Type =
    PythonNumpy::DenseMatrix_Type<T, CARTESIAN_SIZE, CARTESIAN_SIZE>;

/* make Rotation Matrix */

/**
 * @brief Creates a RotationMatrix_Type initialized to the identity matrix.
 *
 * This function constructs a RotationMatrix_Type object representing a
 * 3x3 identity matrix, which is commonly used as a starting point for
 * rotation matrices in 3D space.
 *
 * @tparam T The type of the matrix components (e.g., float, double).
 * @return A RotationMatrix_Type object initialized to the identity matrix.
 */
template <typename T>
inline auto make_RotationMatrix(void) -> RotationMatrix_Type<T> {

  return PythonNumpy::make_DenseMatrix<CARTESIAN_SIZE, CARTESIAN_SIZE>(
      static_cast<T>(1), static_cast<T>(0), static_cast<T>(0),
      static_cast<T>(0), static_cast<T>(1), static_cast<T>(0),
      static_cast<T>(0), static_cast<T>(0), static_cast<T>(1));
}

/**
 * @brief Integrates a quaternion using angular velocity (gyroscope) data with a
 * first-order approximation.
 *
 * This function computes the next orientation quaternion by integrating the
 * current quaternion `q` with the angular velocity vector `omega` over a time
 * step `time_step`. The integration uses a first-order (Euler) approximation of
 * quaternion derivative. The resulting quaternion is normalized with a minimum
 * division threshold `division_min` to avoid division by very small numbers.
 *
 * @tparam T Numeric type (e.g., float, double).
 * @param omega Angular velocity vector (gyroscope data) as an Omega_Type<T>.
 * @param q Current orientation quaternion as a Quaternion_Type<T>.
 * @param time_step Time step over which to integrate.
 * @param division_min Minimum value used during normalization to prevent
 * division by zero.
 * @return Quaternion_Type<T> The integrated and normalized quaternion.
 *
 * Integrates 3-axis angular velocity using an approximate
 * formula to obtain a quaternion.
 * The angular velocity is the angular velocity around each axis of XYZ, and
 * XYZ axis is assumed to be right-handed.
 * The quaternion is a quaternion representing the posture angle.
 * q.w ^ 2 + q.x ^ 2 + q.y ^ 2 + q.z ^ 2 = 1 must be satisfied.
 * The initial posture is q.w = 1, q.x = 0, q.y = 0, q.z = 0 by default.
 */
template <typename T>
inline auto integrate_gyro_approximately(const Omega_Type<T> &omega,
                                         const Quaternion_Type<T> &q,
                                         const T &time_step,
                                         const T &division_min)
    -> Quaternion_Type<T> {

  auto out_q = Quaternion_Type<T>::identity();

  out_q.w = (-(omega.x * q.x) - omega.y * q.y - omega.z * q.z) *
                static_cast<T>(0.5) * time_step +
            q.w;
  out_q.x = (omega.x * q.w + omega.z * q.y - omega.y * q.z) *
                static_cast<T>(0.5) * time_step +
            q.x;
  out_q.y = (omega.y * q.w - omega.z * q.x + omega.x * q.z) *
                static_cast<T>(0.5) * time_step +
            q.y;
  out_q.z = (omega.z * q.w + omega.y * q.x - omega.x * q.y) *
                static_cast<T>(0.5) * time_step +
            q.z;

  return out_q.normalized(division_min);
}

/**
 * @brief Integrates a quaternion using angular velocity (gyroscope) data with a
 * strict approximation.
 *
 * This function computes the next orientation quaternion by integrating the
 * current quaternion `q` with the angular velocity vector `omega` over a time
 * step `time_step`. The integration uses a strict approximation of quaternion
 * derivative, which is more accurate than the first-order approximation. The
 * resulting quaternion is normalized with a minimum division threshold
 * `division_min` to avoid division by very small numbers.
 *
 * @tparam T Numeric type (e.g., float, double).
 * @param omega Angular velocity vector (gyroscope data) as an Omega_Type<T>.
 * @param q Current orientation quaternion as a Quaternion_Type<T>.
 * @param time_step Time step over which to integrate.
 * @param division_min Minimum value used during normalization to prevent
 * division by zero.
 * @return Quaternion_Type<T> The integrated and normalized quaternion.
 *
 * Integrates 3-axis angular velocity using a strict formula to obtain a
 * quaternion.
 * The angular velocity is the angular velocity around each axis of XYZ, and
 * XYZ axis is assumed to be right-handed.
 * The quaternion is a quaternion representing the posture angle.
 * q.w ^ 2 + q.x ^ 2 + q.y ^ 2 + q.z ^ 2 = 1 must be satisfied.
 * The initial posture is q.w = 1, q.x = 0, q.y = 0, q.z = 0 by default.
 */
template <typename T>
inline auto integrate_gyro(const Omega_Type<T> &omega,
                           const Quaternion_Type<T> &q, const T &time_step,
                           const T &division_min) -> Quaternion_Type<T> {

  auto out_q = Quaternion_Type<T>::identity();

  auto omega_norm_inv = Base::Math::rsqrt(
      omega.x * omega.x + omega.y * omega.y + omega.z * omega.z, division_min);

  auto omega_norm_half_step = static_cast<T>(0.5) * time_step / omega_norm_inv;

  T w_sin = static_cast<T>(0);
  T w_cos = static_cast<T>(0);

  Base::Math::sincos(omega_norm_half_step, w_sin, w_cos);

  auto q_omega = Quaternion_Type<T>(w_cos, omega.x * omega_norm_inv * w_sin,
                                    omega.y * omega_norm_inv * w_sin,
                                    omega.z * omega_norm_inv * w_sin);

  out_q = q * q_omega;

  return out_q.normalized(division_min);
}

/**
 * @brief Converts a quaternion to a rotation matrix.
 *
 * This function converts a quaternion representation of orientation into a
 * 3x3 rotation matrix. The conversion is based on the mathematical
 * representation of quaternions and their relationship to rotation matrices.
 *
 * @tparam T The type of the quaternion components (e.g., float, double).
 * @param q The quaternion to be converted.
 * @return A RotationMatrix_Type<T> representing the rotation matrix.
 */
template <typename T>
inline auto as_rotation_matrix(const Quaternion_Type<T> &q)
    -> RotationMatrix_Type<T> {

  auto R = make_RotationMatrix<T>();

  R.template set<P_X, P_X>(q.w * q.w + q.x * q.x - q.y * q.y - q.z * q.z);
  R.template set<P_X, P_Y>((q.x * q.y - q.w * q.z) * static_cast<T>(2));
  R.template set<P_X, P_Z>((q.x * q.z + q.w * q.y) * static_cast<T>(2));

  R.template set<P_Y, P_X>((q.x * q.y + q.w * q.z) * static_cast<T>(2));
  R.template set<P_Y, P_Y>(q.w * q.w - q.x * q.x + q.y * q.y - q.z * q.z);
  R.template set<P_Y, P_Z>((q.y * q.z - q.w * q.x) * static_cast<T>(2));

  R.template set<P_Z, P_X>((q.x * q.z - q.w * q.y) * static_cast<T>(2));
  R.template set<P_Z, P_Y>((q.y * q.z + q.w * q.x) * static_cast<T>(2));
  R.template set<P_Z, P_Z>(q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z);

  return R;
}

/**
 * @brief Converts a quaternion to Euler angles (roll, pitch, yaw).
 *
 * This function converts a quaternion representation of orientation into
 * Euler angles. The conversion is based on the mathematical representation
 * of quaternions and their relationship to Euler angles.
 *
 * @tparam T The type of the quaternion components (e.g., float, double).
 * @param q The quaternion to be converted.
 * @return An EulerAngle_Type<T> representing the Euler angles (roll, pitch,
 * yaw).
 */
template <typename T>
inline auto as_euler_angles(const Quaternion_Type<T> &q) -> EulerAngle_Type<T> {

  auto euler_angles =
      make_EulerAngle(static_cast<T>(0), static_cast<T>(0), static_cast<T>(0));

  auto qw_qw = q.w * q.w;
  auto qx_qx = q.x * q.x;
  auto qy_qy = q.y * q.y;
  auto qz_qz = q.z * q.z;

  euler_angles.x =
      Base::Math::atan(static_cast<T>(2) * (q.y * q.z + q.w * q.x) /
                       (qw_qw - qx_qx - qy_qy + qz_qz));

  euler_angles.y =
      -Base::Math::asin(static_cast<T>(2) * (q.x * q.z - q.w * q.y));

  euler_angles.z =
      Base::Math::atan(static_cast<T>(2) * (q.x * q.y + q.w * q.z) /
                       (qw_qw + qx_qx - qy_qy - qz_qz));

  return euler_angles;
}

} // namespace PythonQuaternion

#endif // __PYTHON_QUATERNION_ROTATION_HPP__
