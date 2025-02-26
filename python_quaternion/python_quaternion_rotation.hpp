#ifndef __PYTHON_QUATERNION_ROTATION_HPP__
#define __PYTHON_QUATERNION_ROTATION_HPP__

#include "python_numpy.hpp"
#include "python_quaternion_base.hpp"

namespace PythonQuaternion {

constexpr std::size_t CARTESIAN_SIZE = 3;
constexpr std::size_t P_X = 0;
constexpr std::size_t P_Y = 1;
constexpr std::size_t P_Z = 2;

constexpr std::size_t E_ROLL = 0;
constexpr std::size_t E_PITCH = 1;
constexpr std::size_t E_YAW = 2;

/* Direction Vector Type */
template <typename T>
using DirectionVector_Type =
    PythonNumpy::DenseMatrix_Type<T, CARTESIAN_SIZE, 1>;

/* make Direction Vector */
template <typename T>
inline auto make_DirectionVector(const T &x, const T &y, const T &z)
    -> DirectionVector_Type<T> {

  return PythonNumpy::make_DenseMatrix<CARTESIAN_SIZE, 1>(x, y, z);
}

/* Quaternion from rotation angle and direction vector */
template <typename T>
inline auto
q_from_rotation_vector(const T &theta,
                       const DirectionVector_Type<T> &direction_vector,
                       const T &division_min) -> Quaternion_Type<T> {

  T direction_vector_sq = direction_vector.template get<P_X, 0>() *
                              direction_vector.template get<P_X, 0>() +
                          direction_vector.template get<P_Y, 0>() *
                              direction_vector.template get<P_Y, 0>() +
                          direction_vector.template get<P_Z, 0>() *
                              direction_vector.template get<P_Z, 0>();

  T direction_vector_norm_inv = static_cast<T>(0);

  direction_vector_norm_inv =
      Base::Math::rsqrt(direction_vector_sq, division_min);

  auto half_theta = theta * static_cast<T>(0.5);
  auto sin_half_theta = Base::Math::sin(half_theta);

  T norm_inv_sin_theta = direction_vector_norm_inv * sin_half_theta;

  return Quaternion_Type<T>(
      Base::Math::cos(half_theta),
      direction_vector.template get<P_X, 0>() * norm_inv_sin_theta,
      direction_vector.template get<P_Y, 0>() * norm_inv_sin_theta,
      direction_vector.template get<P_Z, 0>() * norm_inv_sin_theta);
}

/* Omega Type */
template <typename T>
using Omega_Type = PythonNumpy::DenseMatrix_Type<T, CARTESIAN_SIZE, 1>;

/* make Omega */
template <typename T>
inline auto make_Omega(const T &x, const T &y, const T &z) -> Omega_Type<T> {

  return PythonNumpy::make_DenseMatrix<CARTESIAN_SIZE, 1>(x, y, z);
}

/* Euler Angle Type */
template <typename T>
using EulerAngle_Type = PythonNumpy::DenseMatrix_Type<T, CARTESIAN_SIZE, 1>;

/* make Euler Angle */
template <typename T>
inline auto make_EulerAngle(const T &roll, const T &pitch, const T &yaw)
    -> EulerAngle_Type<T> {

  return PythonNumpy::make_DenseMatrix<CARTESIAN_SIZE, 1>(roll, pitch, yaw);
}

/* Rotation Matrix Type */
template <typename T>
using RotationMatrix_Type =
    PythonNumpy::DenseMatrix_Type<T, CARTESIAN_SIZE, CARTESIAN_SIZE>;

/* make Rotation Matrix */
template <typename T>
inline auto make_RotationMatrix(void) -> RotationMatrix_Type<T> {

  return PythonNumpy::make_DenseMatrix<CARTESIAN_SIZE, CARTESIAN_SIZE>(
      static_cast<T>(1), static_cast<T>(0), static_cast<T>(0),
      static_cast<T>(0), static_cast<T>(1), static_cast<T>(0),
      static_cast<T>(0), static_cast<T>(0), static_cast<T>(1));
}

/***********************************************************************
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
***********************************************************************/
template <typename T>
inline auto integrate_gyro_approximately(const Omega_Type<T> &omega,
                                         const Quaternion_Type<T> &q,
                                         const T &time_step,
                                         const T &division_min)
    -> Quaternion_Type<T> {

  auto out_q = Quaternion_Type<T>::identity();

  out_q.w = (-(omega.template get<P_X, 0>() * q.x) -
             omega.template get<P_Y, 0>() * q.y -
             omega.template get<P_Z, 0>() * q.z) *
                static_cast<T>(0.5) * time_step +
            q.w;
  out_q.x =
      (omega.template get<P_X, 0>() * q.w + omega.template get<P_Z, 0>() * q.y -
       omega.template get<P_Y, 0>() * q.z) *
          static_cast<T>(0.5) * time_step +
      q.x;
  out_q.y =
      (omega.template get<P_Y, 0>() * q.w - omega.template get<P_Z, 0>() * q.x +
       omega.template get<P_X, 0>() * q.z) *
          static_cast<T>(0.5) * time_step +
      q.y;
  out_q.z =
      (omega.template get<P_Z, 0>() * q.w + omega.template get<P_Y, 0>() * q.x -
       omega.template get<P_X, 0>() * q.y) *
          static_cast<T>(0.5) * time_step +
      q.z;

  return out_q.normalized(division_min);
}

/***********************************************************************
# Integrates 3-axis angular velocity to obtain a quaternion (strict)

Arguments:
  omega: 3-axis angular velocity [rad/s]
  q: Quaternion one step before
  time_step: Integration time step [s]

Returns:
  q: Quaternion

Details:
  Integrates 3-axis angular velocity using a strict formula to obtain a
  quaternion. The angular velocity is the angular velocity around each axis of
  XYZ, and XYZ axis is assumed to be right-handed. The quaternion is a
quaternion representing the posture angle. q.w ^ 2 + q.x ^ 2 + q.y ^ 2 + q.z ^
2= 1 must be satisfied. The initial posture is q.w = 1, q.x = 0, q.y = 0, q.z =
0 by default.
 ***********************************************************************/
template <typename T>
inline auto integrate_gyro(const Omega_Type<T> &omega,
                           const Quaternion_Type<T> &q, const T &time_step,
                           const T &division_min) -> Quaternion_Type<T> {

  auto out_q = Quaternion_Type<T>::identity();

  auto omega_norm_inv = Base::Math::rsqrt(
      omega.template get<P_X, 0>() * omega.template get<P_X, 0>() +
          omega.template get<P_Y, 0>() * omega.template get<P_Y, 0>() +
          omega.template get<P_Z, 0>() * omega.template get<P_Z, 0>(),
      division_min);

  auto omega_norm_half_step = static_cast<T>(0.5) * time_step / omega_norm_inv;

  T w_sin = static_cast<T>(0);
  T w_cos = static_cast<T>(0);

  Base::Math::sincos(omega_norm_half_step, w_sin, w_cos);

  auto q_omega = Quaternion_Type<T>(
      w_cos, omega.template get<P_X, 0>() * omega_norm_inv * w_sin,
      omega.template get<P_Y, 0>() * omega_norm_inv * w_sin,
      omega.template get<P_Z, 0>() * omega_norm_inv * w_sin);

  out_q = q * q_omega;

  return out_q.normalized(division_min);
}

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

template <typename T>
inline auto as_euler_angles(const Quaternion_Type<T> &q) -> EulerAngle_Type<T> {

  auto euler_angles =
      make_EulerAngle(static_cast<T>(0), static_cast<T>(0), static_cast<T>(0));

  auto qw_qw = q.w * q.w;
  auto qx_qx = q.x * q.x;
  auto qy_qy = q.y * q.y;
  auto qz_qz = q.z * q.z;

  euler_angles.template set<E_ROLL, 0>(
      Base::Math::atan(static_cast<T>(2) * (q.y * q.z + q.w * q.x) /
                       (qw_qw - qx_qx - qy_qy + qz_qz)));

  euler_angles.template set<E_PITCH, 0>(
      -Base::Math::asin(static_cast<T>(2) * (q.x * q.z - q.w * q.y)));

  euler_angles.template set<E_YAW, 0>(
      Base::Math::atan(static_cast<T>(2) * (q.x * q.y + q.w * q.z) /
                       (qw_qw + qx_qx - qy_qy - qz_qz)));

  return euler_angles;
}

} // namespace PythonQuaternion

#endif // __PYTHON_QUATERNION_ROTATION_HPP__
