#include <type_traits>
#include <iostream>
#include <cmath>

#include "python_quaternion.hpp"

#include "MCAP_tester.hpp"


using namespace Tester;
using namespace PythonQuaternion;


template <typename T>
void check_python_quaternion_base(void) {
    using namespace PythonNumpy;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-3);
    //constexpr T NEAR_LIMIT_SOFT = 1.0e-2F;

    /* 単位変換 */
    auto degree = radian_to_degree(static_cast<T>(0.1));

    tester.expect_near(degree, static_cast<T>(5.729577951308232), NEAR_LIMIT_STRICT,
        "check radian_to_degree ");

    auto radian = degree_to_radian(static_cast<T>(5.729577951308232));

    tester.expect_near(radian, static_cast<T>(0.1), NEAR_LIMIT_STRICT,
        "check degree_to_radian ");

    /* クオータニオン定義 */
    Quaternion<T> q1;
    q1.w = static_cast<T>(1);
    q1.x = static_cast<T>(2);
    q1.y = static_cast<T>(3);
    q1.z = static_cast<T>(4);

    Quaternion<T> q1_copy = q1;
    Quaternion<T> q1_move(q1_copy);
    q1 = std::move(q1_move);

    tester.expect_near(q1.w, static_cast<T>(1), NEAR_LIMIT_STRICT,
        "check quaternion copy and move ");
    tester.expect_near(q1.x, static_cast<T>(2), NEAR_LIMIT_STRICT,
        "check quaternion copy and move ");
    tester.expect_near(q1.y, static_cast<T>(3), NEAR_LIMIT_STRICT,
        "check quaternion copy and move ");
    tester.expect_near(q1.z, static_cast<T>(4), NEAR_LIMIT_STRICT,
        "check quaternion copy and move ");

    Quaternion_Type<T> q2 = make_quaternion(
        static_cast<T>(5),
        static_cast<T>(6),
        static_cast<T>(7),
        static_cast<T>(8)
    );

    /* 単位クオータニオン */
    auto q1_identity = Quaternion<T>::identity();

    tester.expect_near(q1_identity.w, static_cast<T>(1), NEAR_LIMIT_STRICT,
        "check quaternion identity ");
    tester.expect_near(q1_identity.x, static_cast<T>(0), NEAR_LIMIT_STRICT,
        "check quaternion identity ");
    tester.expect_near(q1_identity.y, static_cast<T>(0), NEAR_LIMIT_STRICT,
        "check quaternion identity ");
    tester.expect_near(q1_identity.z, static_cast<T>(0), NEAR_LIMIT_STRICT,
        "check quaternion identity ");

    /* 回転軸ベクトルからクオータニオン */
    T theta = static_cast<T>(Base::Math::PI) / static_cast<T>(4);
    auto direction_vector = make_DirectionVector(
        static_cast<T>(1),
        static_cast<T>(2),
        static_cast<T>(3)
    );

    auto q_r = q_from_rotation_vector(theta, direction_vector, static_cast<T>(1.0e-10));

    tester.expect_near(q_r.w, static_cast<T>(0.923879532511287), NEAR_LIMIT_STRICT,
        "check q_from_rotation_vector ");
    tester.expect_near(q_r.x, static_cast<T>(0.102276449393203), NEAR_LIMIT_STRICT,
        "check q_from_rotation_vector ");
    tester.expect_near(q_r.y, static_cast<T>(0.204552898786406), NEAR_LIMIT_STRICT,
        "check q_from_rotation_vector ");
    tester.expect_near(q_r.z, static_cast<T>(0.306829348179609), NEAR_LIMIT_STRICT,
        "check q_from_rotation_vector ");

    /* クオータニオン和 */
    auto q1_add_q2 = q1 + q2;

    tester.expect_near(q1_add_q2.w, static_cast<T>(6), NEAR_LIMIT_STRICT,
        "check quaternion addition ");
    tester.expect_near(q1_add_q2.x, static_cast<T>(8), NEAR_LIMIT_STRICT,
        "check quaternion addition ");
    tester.expect_near(q1_add_q2.y, static_cast<T>(10), NEAR_LIMIT_STRICT,
        "check quaternion addition ");
    tester.expect_near(q1_add_q2.z, static_cast<T>(12), NEAR_LIMIT_STRICT,
        "check quaternion addition ");

    /* クオータニオン差 */
    auto q1_sub_q2 = q1 - q2;

    tester.expect_near(q1_sub_q2.w, static_cast<T>(-4), NEAR_LIMIT_STRICT,
        "check quaternion subtraction ");
    tester.expect_near(q1_sub_q2.x, static_cast<T>(-4), NEAR_LIMIT_STRICT,
        "check quaternion subtraction ");
    tester.expect_near(q1_sub_q2.y, static_cast<T>(-4), NEAR_LIMIT_STRICT,
        "check quaternion subtraction ");
    tester.expect_near(q1_sub_q2.z, static_cast<T>(-4), NEAR_LIMIT_STRICT,
        "check quaternion subtraction ");

    /* クオータニオン積 */
    auto q1_mul_q2 = q1 * q2;

    tester.expect_near(q1_mul_q2.w, static_cast<T>(-60), NEAR_LIMIT_STRICT,
        "check quaternion product ");
    tester.expect_near(q1_mul_q2.x, static_cast<T>(12), NEAR_LIMIT_STRICT,
        "check quaternion product ");
    tester.expect_near(q1_mul_q2.y, static_cast<T>(30), NEAR_LIMIT_STRICT,
        "check quaternion product ");
    tester.expect_near(q1_mul_q2.z, static_cast<T>(24), NEAR_LIMIT_STRICT,
        "check quaternion product ");

    /* 共役 */
    auto q1_conj = q1.conjugate();

    tester.expect_near(q1_conj.w, static_cast<T>(1), NEAR_LIMIT_STRICT,
        "check quaternion conjugate ");
    tester.expect_near(q1_conj.x, static_cast<T>(-2), NEAR_LIMIT_STRICT,
        "check quaternion conjugate ");
    tester.expect_near(q1_conj.y, static_cast<T>(-3), NEAR_LIMIT_STRICT,
        "check quaternion conjugate ");
    tester.expect_near(q1_conj.z, static_cast<T>(-4), NEAR_LIMIT_STRICT,
        "check quaternion conjugate ");

    /* ノルム */
    auto q1_norm = q1.norm();

    tester.expect_near(q1_norm, static_cast<T>(5.477225575051661), NEAR_LIMIT_STRICT,
        "check quaternion norm ");

    /* 正規化 */
    auto q1_normalized = q1.normalized(static_cast<T>(1e-10));

    tester.expect_near(q1_normalized.w, static_cast<T>(0.182574185835055), NEAR_LIMIT_STRICT,
        "check quaternion normalization ");
    tester.expect_near(q1_normalized.x, static_cast<T>(0.365148371670111), NEAR_LIMIT_STRICT,
        "check quaternion normalization ");
    tester.expect_near(q1_normalized.y, static_cast<T>(0.547722557505166), NEAR_LIMIT_STRICT,
        "check quaternion normalization ");
    tester.expect_near(q1_normalized.z, static_cast<T>(0.730296743340221), NEAR_LIMIT_STRICT,
        "check quaternion normalization ");

    /* クオータニオン除 */
    auto q1_div_q2 = quaternion_divide(q1, q2, static_cast<T>(1e-10));

    tester.expect_near(q1_div_q2.w, static_cast<T>(0.402298850574713), NEAR_LIMIT_STRICT,
        "check quaternion division ");
    tester.expect_near(q1_div_q2.x, static_cast<T>(0.0459770114942529), NEAR_LIMIT_STRICT,
        "check quaternion division ");
    tester.expect_near(q1_div_q2.y, static_cast<T>(0), NEAR_LIMIT_STRICT,
        "check quaternion division ");
    tester.expect_near(q1_div_q2.z, static_cast<T>(0.0919540229885057), NEAR_LIMIT_STRICT,
        "check quaternion division ");


    tester.throw_error_if_test_failed();
}


template <typename T>
void check_python_rotation(void) {
    using namespace PythonNumpy;

    MCAPTester<T> tester;

    constexpr T NEAR_LIMIT_STRICT = std::is_same<T, double>::value ? T(1.0e-5) : T(1.0e-4);
    //constexpr T NEAR_LIMIT_SOFT = 1.0e-2F;

    auto q_1 = Quaternion<T>::identity();
    auto omega_1 = make_Omega(
        static_cast<T>(1),
        static_cast<T>(-1),
        static_cast<T>(1)
    );

    auto q_1_next_approximate = integrate_gyro_approximately(omega_1, q_1, static_cast<T>(0.1), static_cast<T>(1.0e-10));

    tester.expect_near(q_1_next_approximate.w, static_cast<T>(0.996270962773436), NEAR_LIMIT_STRICT,
        "check integrate_gyro_approximately ");
    tester.expect_near(q_1_next_approximate.x, static_cast<T>(0.0498135481386718), NEAR_LIMIT_STRICT,
        "check integrate_gyro_approximately ");
    tester.expect_near(q_1_next_approximate.y, static_cast<T>(-0.0498135481386718), NEAR_LIMIT_STRICT,
        "check integrate_gyro_approximately ");
    tester.expect_near(q_1_next_approximate.z, static_cast<T>(0.0498135481386718), NEAR_LIMIT_STRICT,
        "check integrate_gyro_approximately ");

    auto q_1_next = integrate_gyro(omega_1, q_1, static_cast<T>(0.1), static_cast<T>(1.0e-10));

    tester.expect_near(q_1_next.w, static_cast<T>(0.996252343164141), NEAR_LIMIT_STRICT,
        "check integrate_gyro ");
    tester.expect_near(q_1_next.x, static_cast<T>(0.0499375234333152), NEAR_LIMIT_STRICT,
        "check integrate_gyro ");
    tester.expect_near(q_1_next.y, static_cast<T>(-0.0499375234333152), NEAR_LIMIT_STRICT,
        "check integrate_gyro ");
    tester.expect_near(q_1_next.z, static_cast<T>(0.0499375234333152), NEAR_LIMIT_STRICT,
        "check integrate_gyro ");

    /* 回転行列を作成 */
    auto R_q_1_next = as_rotation_matrix(q_1_next);

    Matrix<DefDense, T, 3, 3> R_q_1_next_answer({
        {static_cast<T>(0.99002498), static_cast<T>(-0.10448826), static_cast<T>(-0.09451324)},
        {static_cast<T>(0.09451324), static_cast<T>(0.99002498), static_cast<T>(-0.10448826)},
        {static_cast<T>(0.10448826), static_cast<T>(0.09451324), static_cast<T>(0.99002498)}
        });

    tester.expect_near(R_q_1_next.matrix.data, R_q_1_next_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check as_rotation_matrix ");

    /* クオータニオンからオイラー角 */
    auto euler_angle = as_euler_angles(q_1_next);

    Matrix<DefDense, T, 3, 1> euler_angle_answer({
        {static_cast<T>(9.51770699027242E-02)},
        {static_cast<T>(-1.04679332455626E-01)},
        {static_cast<T>(9.51770699027242E-02)}
        });

    tester.expect_near(euler_angle.matrix.data, euler_angle_answer.matrix.data, NEAR_LIMIT_STRICT,
        "check as_euler_angles ");



    tester.throw_error_if_test_failed();
}


int main(void) {

    check_python_quaternion_base<double>();

    check_python_quaternion_base<float>();

    check_python_rotation<double>();

    check_python_rotation<float>();


    return 0;
}
