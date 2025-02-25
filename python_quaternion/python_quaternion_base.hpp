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
template <typename T> inline auto radian_to_degree(const T &radian) -> T {
  return radian * static_cast<T>(180) / static_cast<T>(Base::Math::PI);
}

template <typename T> inline auto degree_to_radian(const T &degree) -> T {
  return degree * static_cast<T>(Base::Math::PI) / static_cast<T>(180);
}

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
  static inline auto identity() -> Quaternion<T> {
    return Quaternion<T>(1, 0, 0, 0);
  }

  inline auto conjugate() const -> Quaternion<T> {
    return Quaternion<T>(w, -x, -y, -z);
  }

  inline auto norm() const -> T {
    return Base::Math::sqrt(w * w + x * x + y * y + z * z);
  }

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
template <typename T>
inline auto operator+(const Quaternion<T> &q1, const Quaternion<T> &q2)
    -> Quaternion<T> {
  return Quaternion<T>(q1.w + q2.w, q1.x + q2.x, q1.y + q2.y, q1.z + q2.z);
}

/* Quaternion Subtraction */
template <typename T>
inline auto operator-(const Quaternion<T> &q1, const Quaternion<T> &q2)
    -> Quaternion<T> {
  return Quaternion<T>(q1.w - q2.w, q1.x - q2.x, q1.y - q2.y, q1.z - q2.z);
}

/* Quaternion Product */
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
template <typename T>
inline auto make_quaternion(const T &w, const T &x, const T &y, const T &z)
    -> Quaternion<T> {
  return Quaternion<T>(w, x, y, z);
}

} // namespace PythonQuaternion

#endif // __PYTHON_QUATERNION_BASE_HPP__
