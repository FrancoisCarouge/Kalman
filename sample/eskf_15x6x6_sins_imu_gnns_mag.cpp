/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.3
https://github.com/FrancoisCarouge/Kalman

SPDX-License-Identifier: Unlicense

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org> */

// Add the case where both data are available?

// LESSON: THERE ARE TWO (N) MEASUREMENT MODELs: WITH THEIR OWN H, R, AND
// UPDATE FUNCTION. NEED STRONG TYPE SUPPORT: THE MEASUREMENT COULD BE
// DIFFERENT SIZES, TYPES. COMPOSE?

#include "fcarouge/kalman.hpp"
#include "fcarouge/linalg.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <print>
#include <random>

namespace fcarouge::sample {
namespace {
template <auto Size> using vector = column_vector<double, Size>;
template <auto Row, auto Column> using matrix = matrix<double, Row, Column>;

matrix<3, 3> skew(const vector<3> &v) {
  return matrix<3, 3>{
      {0, -v.z(), v.y()}, {v.z(), 0, -v.x()}, {-v.y(), v.x(), 0}};
}

// Approximated gravity vector.
const vector<3> gravity{0., 0., 9.81};

// Approximated normalized Earth's magnetic field.
const vector<3> magnetic_field_inertial{
    vector<3>{0.5, 0.0, 0.866}.normalized()};

struct SimulationData {
  double t;
  vector<3> true_pos;
  vector<3> true_vel;
  Eigen::Quaterniond true_quat;

  // Sensor readings (Noisy)
  vector<3> imu_acc;
  vector<3> imu_gyro;
  vector<3> mag;
  vector<3> gnss_pos;

  // Availability flags
  bool has_gnss;
  bool has_mag;
};

// TODO: Replace by a data file.
class Simulator {
  std::mt19937 gen_{0}; // std::random_device{}()};
  std::normal_distribution<double> dist_acc_{0.0, 0.2};  // m/s^2 noise
  std::normal_distribution<double> dist_gyr_{0.0, 0.01}; // rad/s noise
  std::normal_distribution<double> dist_gnss_{0.0, 1.5}; // m noise
  std::normal_distribution<double> dist_mag_{0.0, 0.05}; // normalized noise

  // Biases (Constant for sim)
  vector<3> true_ba_{0.1, -0.1, 0.05};
  vector<3> true_bg_{0.01, 0.01, -0.01};

public:
  SimulationData step(double t, double dt) {
    SimulationData data;
    data.t = t;

    // Trajectory: Figure 8 in XY plane, sinusoidal Z
    double freq = 0.2;
    double A = 50.0; // Radius

    // Parametric Position
    data.true_pos =
        vector<3>{A * std::sin(freq * t),
                  A * std::sin(freq * t) * std::cos(freq * t), // Lemniscate-ish
                  10.0 * std::sin(0.5 * t)};

    // Analytic Velocity (Derivative)
    // Simplified for demo: Finite difference for robust "truth" generation
    static vector<3> last_pos = data.true_pos;
    if (t == 0)
      last_pos = data.true_pos;
    data.true_vel = (data.true_pos - last_pos) / dt;
    last_pos = data.true_pos;

    // Analytic Acceleration (Finite diff)
    static vector<3> last_vel = data.true_vel;
    vector<3> true_acc_inertial = (data.true_vel - last_vel) / dt;
    last_vel = data.true_vel;

    // Attitude: Point velocity vector forward + some roll
    if (data.true_vel.norm() > 0.1) {
      vector<3> forward = data.true_vel.normalized();
      vector<3> right = forward.cross(vector<3>::UnitZ()).normalized();
      vector<3> up = right.cross(forward);
      matrix<3, 3> R_true;
      R_true.col(0) = forward;
      R_true.col(1) = right;
      R_true.col(2) = up;
      data.true_quat = Eigen::Quaterniond(R_true);
    } else {
      data.true_quat = Eigen::Quaterniond::Identity();
    }

    // True Gyro (Finite diff of attitude not trivial, approx body rates)
    // For simulation, we just infer required omega to match R change
    // Or simpler: generate sensors from kinematics?
    // Let's use reverse: a_m = R^T(a_inertial - g)
    matrix<3, 3> R_t = data.true_quat.toRotationMatrix().transpose();
    vector<3> a_body =
        R_t * (true_acc_inertial + gravity); // Proper acceleration

    // Add noise & bias to Accel
    data.imu_acc = a_body + true_ba_ +
                   vector<3>{dist_acc_(gen_), dist_acc_(gen_), dist_acc_(gen_)};

    // Gyro: simplified, just add noise to a synthetic omega
    // Calculating true omega from discrete quat steps
    // q_next = q * Exp(w * dt) -> Exp(w*dt) = q_inv * q_next
    static Eigen::Quaterniond last_q = data.true_quat;
    Eigen::Quaterniond dq = last_q.conjugate() * data.true_quat;
    Eigen::AngleAxisd aa(dq);
    vector<3> true_omega = aa.axis() * aa.angle() / dt;
    last_q = data.true_quat;

    data.imu_gyro =
        true_omega + true_bg_ +
        vector<3>{dist_gyr_(gen_), dist_gyr_(gen_), dist_gyr_(gen_)};

    // Magnetometer
    vector<3> mag_body = R_t * magnetic_field_inertial;
    data.mag =
        mag_body + vector<3>{dist_mag_(gen_), dist_mag_(gen_), dist_mag_(gen_)};

    // GNSS
    data.gnss_pos =
        data.true_pos +
        vector<3>{dist_gnss_(gen_), dist_gnss_(gen_), dist_gnss_(gen_)};

    // Rates
    data.has_gnss = (std::fmod(t, 1.0) < dt); // 1Hz
    data.has_mag = (std::fmod(t, 0.1) < dt);  // 10Hz

    return data;
  }
};

// TODO: Best ways to natively support error-state paradigms in the library
// patterns?
void inject_error(auto &nominal, auto &filter) {
  // Inject position error into nominal position.
  nominal.template head<3>() += filter.x().template head<3>();

  // Inject velocity error into nominal velocity.
  nominal.template segment<3>(3) += filter.x().template segment<3>(3);

  // Inject attitude error into nominal attitude.
  vector<3> dtheta{filter.x().template segment<3>(6)};
  Eigen::Quaterniond dq; // dq = [1, 0.5*dtheta] for small errors
  dq.w() = 1.0;
  dq.vec() = 0.5 * dtheta;
  dq.normalize();
  nominal.template segment<4>(6) = // Nominal = Nominal * Error
      (Eigen::Quaterniond{nominal[9], nominal[6], nominal[7], nominal[8]} * dq)
          .normalized()
          .coeffs();

  // Inject bias acceleration error into nominal bias acceleration.
  nominal.template segment<3>(10) += filter.x().template segment<3>(9);

  // Inject bias gyroscope error into nominal bias gyroscope.
  nominal.template segment<3>(13) += filter.x().template segment<3>(12);

  // Reset error
  filter.x(vector<15>::Zero());
}

void update_gnss(auto &nominal, auto &filter, const vector<3> &p_measured) {
  matrix<3, 3> r{matrix<3, 3>::Identity() * 2.25}; // 1.5m std dev squared

  // Measurement model: y = p + noise
  // H matrix selects position from error state
  matrix<3, 15> h{matrix<3, 15>::Zero()};
  h.block<3, 3>(0, 0) = matrix<3, 3>::Identity();

  vector<3> y{p_measured - nominal.template head<3>()};
  matrix<3, 3> s{h * filter.p() * h.transpose() + r};
  matrix<15, 3> k{filter.p() * h.transpose() * s.inverse()};

  // Error State Update
  filter.x(k * y);
  matrix<15, 15> I_KH{matrix<15, 15>::Identity() - k * h};
  filter.p(I_KH * filter.p() * I_KH.transpose() +
           k * r * k.transpose()); // Joseph form

  inject_error(nominal, filter);
}

void update_mag(auto &nominal, auto &filter, const vector<3> &m_measured) {
  matrix<3, 3> r{matrix<3, 3>::Identity() * 0.01};

  // Predicted measurement in body frame: m_b = R(q)^T * m_inertial
  matrix<3, 3> R =
      Eigen::Quaterniond{nominal[9], nominal[6], nominal[7], nominal[8]}
          .toRotationMatrix();
  vector<3> m_predicted{R.transpose() * magnetic_field_inertial};

  vector<3> y{m_measured - m_predicted};

  // Jacobian H w.r.t angular error theta
  // m_b_new approx m_b_old + [m_b_old]x * delta_theta
  // So H = [m_b]x
  matrix<3, 15> h{matrix<3, 15>::Zero()};
  h.block<3, 3>(0, 6) = skew(m_predicted);
  matrix<3, 3> s{h * filter.p() * h.transpose() + r};
  matrix<15, 3> k{filter.p() * h.transpose() * s.inverse()};

  // Error State Update
  filter.x(k * y);
  // Simple covariance update for demo (Joseph form preferred for stability)
  // filter.p((matrix<15, 15>::Identity() - k * h) * filter.p());
  matrix<15, 15> I_KH{matrix<15, 15>::Identity() - k * h};
  filter.p(I_KH * filter.p() * I_KH.transpose() +
           k * r * k.transpose()); // Joseph form

  inject_error(nominal, filter);
}

//! @brief Strapdown Inertial Navigation System (SINS) filter.
//!
//! @details Error-State Kalman Filter (ESKF) of the standard Strapdown Inertial
//! Navigation System (SINS) formulation using data from an Inertial Measurement
//! Unit (IMU), Global Navigation Satellite System (GNNS), and magnetometer. 16
//! nominal states (3D position, 3D velocity, attitude quaternion, 3D
//! accelerometer bias, and 3D gyroscope bias) computed from the estimation of
//! 15 error states (???). The GNSS and magnetometer data update, correct the
//! position and heading, respectively predicted by the integration of the IMU
//! data kinematics.
//!
//! @see Quaternion kinematics for the error-state Kalman filter by Joan Sola,
//! November 8, 2017
//!
//! @example eskf_15x6x6_sins_imu_gnns_mag.cpp
[[maybe_unused]] auto sample{[] {
  // The nominal, true, physical state: [position 3D, velocity 3d, attitude
  // quaternion, accelerometer 3D bias, gyroscopic 3D]: [px, py, pz, vx, vy,
  // vz, qx, qy, qz, qw, baccx, baccy, baccz, bgyrx, bgyry, bgyrz]
  vector<16> nominal{vector<16>::Zero()};
  // TODO: Eigen::Quaterniond are constructed {w, x, y, z} but memory ordered
  // [x, y, z, w]. An abstraction is required for type safety.
  nominal[9] = 1.; // qw = 1

  kalman filter{

      // The estimated error state:
      // [position 3D error, velocity 3D error, Theta 3D angular error,
      // accelerometer 3D bias error, gyroscopic 3D bias error]:
      // [px, py, pz, vx, vy, vz, θx, θy, θz, baccx, baccy, baccz, bgyrx, bgyry,
      // bgyrz]
      state{vector<15>{0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                       0.}},

      // The measured error output Z is GNSS and magnetometer data.
      // [px, py, pz, mx, my, mz]
      output<vector<6>>,

      // The control input U is the IMU accelerometer and gyroscope data.
      input<vector<6>>,

      // The initial error estimate uncertainty P is a high uncertainty in
      // position and a low uncertainy in accelerometer bias.
      estimate_uncertainty{matrix<15, 15>{
          {10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
          {0., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
          {0., 0., 10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
          {0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
          {0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
          {0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.},
          {0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.},
          {0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.},
          {0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.},
          {0., 0., 0., 0., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0., 0.},
          {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.1, 0., 0., 0., 0.},
          {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.1, 0., 0., 0.},
          {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.},
          {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.},
          {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.}}},

      // The tunable error process uncertainty Q is uniform velocity noise
      // from acceleration noise, angle noise from gyroscope noise, and
      // biases noise.
      process_uncertainty{
          []([[maybe_unused]] const vector<15> &x, const double &dt) {
            constexpr double accelerometer_noise_density{0.1};
            constexpr double gyroscope_noise_density{0.01};
            constexpr double accelerometer_bias_random_walk{0.001};
            constexpr double gyroscope_bias_random_walk{0.000'1};
            matrix<15, 15> q{kalman_internal::zero<matrix<15, 15>>};
            q[3, 3] = accelerometer_noise_density * dt * dt;
            q[4, 4] = accelerometer_noise_density * dt * dt;
            q[5, 5] = accelerometer_noise_density * dt * dt;
            q[6, 6] = gyroscope_noise_density * dt * dt;
            q[7, 7] = gyroscope_noise_density * dt * dt;
            q[8, 8] = gyroscope_noise_density * dt * dt;
            q[9, 9] = accelerometer_bias_random_walk * dt;
            q[10, 10] = accelerometer_bias_random_walk * dt;
            q[11, 11] = accelerometer_bias_random_walk * dt;
            q[12, 12] = gyroscope_bias_random_walk * dt;
            q[13, 13] = gyroscope_bias_random_walk * dt;
            q[14, 14] = gyroscope_bias_random_walk * dt;

            return q;
          }},

      // The output uncertainty R is 1.5 m standard deviation position and
      // 0.1 (?) standard deviation (magneto?).
      output_uncertainty{matrix<6, 6>{{2.25, 0., 0., 0., 0., 0.},
                                      {0., 2.25, 0., 0., 0., 0.},
                                      {0., 0., 2.25, 0., 0., 0.},
                                      {0., 0., 0., 0.01, 0., 0.},
                                      {0., 0., 0., 0., 0.01, 0.},
                                      {0., 0., 0., 0., 0., 0.01}}},

      // The output model H 6x15 matrix selects the position measured
      // from the GNSS and the Jacobian for the angular error theta from
      // the magnetometer heading.
      output_model{[&nominal]([[maybe_unused]] const vector<15> &x) {
        // Measurement model: y = p + noise
        // H matrix selects position from error state
        matrix<6, 15> h{matrix<6, 15>::Zero()};
        h.block<3, 3>(0, 0) = matrix<3, 3>::Identity();

        // Predicted measurement in body frame: m_b = R(q)^T *
        // m_inertial
        matrix<3, 3> R =
            Eigen::Quaterniond{nominal[9], nominal[6], nominal[7], nominal[8]}
                .toRotationMatrix();
        vector<3> m_predicted = R.transpose() * magnetic_field_inertial;

        // Jacobian H w.r.t angular error theta
        // m_b_new approx m_b_old + [m_b_old]x * delta_theta
        // So H = [m_b]x
        // AI says this should be: -skew ?
        h.block<3, 3>(3, 6) = skew(m_predicted);

        return h;
      }},

      // The state transition F 15x15 matrix is the error-state
      // Jacobian.
      state_transition{[&nominal]([[maybe_unused]] const vector<15> &x,
                                  [[maybe_unused]] const vector<6> &u,
                                  [[maybe_unused]] const double &dt) {
        vector<3> a_m = u.head<3>();
        vector<3> w_m = u.tail<3>();
        // 1. Correct IMU measurements with current bias estimates
        [[maybe_unused]] vector<3> a_unbiased = a_m - nominal.segment<3>(10);
        [[maybe_unused]] vector<3> w_unbiased = w_m - nominal.segment<3>(13);

        // 2. Nominal State Kinematics
        [[maybe_unused]] matrix<3, 3> R =
            Eigen::Quaterniond{nominal[9], nominal[6], nominal[7], nominal[8]}
                .toRotationMatrix();

        // Biases are constant in prediction (Random Walk modeled in P)

        // 3. Error State Jacobian (f_x)
        Eigen::Matrix<double, 15, 15> f =
            Eigen::Matrix<double, 15, 15>::Identity();

        // Position blocks
        f.block<3, 3>(0, 3) = matrix<3, 3>::Identity() * dt;

        // Velocity blocks
        // dv/dtheta = -R * [a]x
        matrix<3, 3> a_skew; // USE SKEW?
        a_skew << 0, -a_unbiased.z(), a_unbiased.y(), a_unbiased.z(), 0,
            -a_unbiased.x(), -a_unbiased.y(), a_unbiased.x(), 0;
        f.block<3, 3>(3, 6) = -R * a_skew * dt;
        f.block<3, 3>(3, 9) = -R * dt; // dv/dba

        // Angle blocks
        // dtheta/dtheta = Transpose(Rot(w*dt)) approx I - [w*dt]x
        // For small dt, often approximated as Identity, but let's be
        // precise.
        f.block<3, 3>(6, 6) = matrix<3, 3>::Identity() - skew(w_unbiased * dt);
        f.block<3, 3>(6, 12) = -matrix<3, 3>::Identity() * dt;

        return f;
      }},

      observation{[&nominal]([[maybe_unused]] const vector<15> &x) {
        vector<6> o{vector<6>::Zero()};
        o.head<3>() = nominal.head<3>();

        matrix<3, 3> R =
            Eigen::Quaterniond{nominal[9], nominal[6], nominal[7], nominal[8]}
                .toRotationMatrix();
        o.tail<3>() = R.transpose() * magnetic_field_inertial;

        return o;
      }},

      transition{[&nominal]([[maybe_unused]] const vector<15> &x,
                            const vector<6> &u, const double &dt) {
        vector<3> a_m = u.head<3>();
        vector<3> w_m = u.tail<3>();
        // 1. Correct IMU measurements with current bias estimates
        vector<3> a_unbiased = a_m - nominal.segment<3>(10);
        vector<3> w_unbiased = w_m - nominal.segment<3>(13);

        // 2. Nominal State Kinematics
        matrix<3, 3> R =
            Eigen::Quaterniond{nominal[9], nominal[6], nominal[7], nominal[8]}
                .toRotationMatrix();

        // Position update: p = p + v*dt + 0.5*(R*a - g)*dt^2
        vector<3> acc_world = R * a_unbiased - gravity;
        nominal.head<3>() +=
            nominal.segment<3>(3) * dt + 0.5 * acc_world * dt * dt;

        // Velocity update: v = v + (R*a - g)*dt
        nominal.segment<3>(3) += acc_world * dt;

        // Attitude update: q = q * Exp(w*dt)
        // Using 0th order integration for quaternion (small angles)
        // AI: If norm() is tiny but > 0, normalization explodes noise.
        // Check?
        Eigen::AngleAxisd rotation_vector(w_unbiased.norm() * dt,
                                          w_unbiased.normalized());
        Eigen::Quaterniond delta_q;
        if (w_unbiased.norm() > 1e-8) {
          delta_q = Eigen::Quaterniond(rotation_vector);
        } else {
          delta_q = Eigen::Quaterniond::Identity();
        }
        nominal.segment<4>(6) = (Eigen::Quaterniond{nominal[9], nominal[6],
                                                    nominal[7], nominal[8]} *
                                 delta_q)
                                    .normalized()
                                    .coeffs();

        return vector<15>::Zero(); // The error state kinematics
                                   // propagate as zero.
      }},

      // The filter predicts over time (seconds).
      prediction_types<double>};

  // Simulate the filter with:
  // * 100Hz IMU 3D acceleration and gyroscope synchronous input,
  // * 10Hz magnetometer output measurement,
  // * 1Hz GNSS output measurement.

  Simulator sim;
  double t = 0;
  double dt = 0.01; // 100Hz IMU
  double duration = 30.0;

  // Simulation Loop
  std::cout << std::format("{:<10} {:<25} {:<25} {:<15}\n", "Time(s)",
                           "Pos Error (m)", "Att Error (deg)", "Status");
  std::cout << std::string(80, '-') << "\n";

  while (t < duration) {
    auto data = sim.step(t, dt);

    filter.predict(dt, data.imu_acc.x(), data.imu_acc.y(), data.imu_acc.z(),
                   data.imu_gyro.x(), data.imu_gyro.y(), data.imu_gyro.z());

    // Can we explain the difference?
    // if (data.has_gnss && data.has_mag) {
    //   filter.update(data.gnss_pos.x(), data.gnss_pos.y(), data.gnss_pos.z(),
    //                 data.mag.x(), data.mag.y(), data.mag.z());
    //   inject_error(nominal, filter);
    // } else
    if (data.has_gnss) {
      update_gnss(nominal, filter, data.gnss_pos);
    }
    // else
    if (data.has_mag) {
      update_mag(nominal, filter, data.mag);
    }

    // 3. Analysis / Logging (1Hz)
    if (std::fmod(t, 1.0) < dt) {
      double pos_err = (nominal.head<3>() - data.true_pos).norm();

      // Quaternion angular distance
      Eigen::Quaterniond q_err_q =
          Eigen::Quaterniond{nominal[9], nominal[6], nominal[7], nominal[8]}
              .conjugate() *
          data.true_quat;
      double att_err_deg =
          2.0 * std::atan2(q_err_q.vec().norm(), std::abs(q_err_q.w())) *
          180.0 / std::numbers::pi;

      std::cout << std::format("{:<10.2f} {:<25.8f} {:<25.8f} ", t, pos_err,
                               att_err_deg);

      if (pos_err < 5.0 && att_err_deg < 5.0) {
        std::cout << "[CONVERGED]\n";
      } else {
        std::cout << "[ALIGNING]\n";
      }
    }

    t += dt;
  }

  // Final Report
  std::cout << "\nSimulation Complete.\n";
  std::cout << "Final Bias Est (Acc): " << nominal.segment<3>(10).transpose()
            << "\n";
  std::cout << "True Bias (Acc)     : 0.1 -0.1 0.05\n";

  // assert(nominal[10] == -0.3427876247455756142);
  // assert(nominal[11] == 0.4485162499302509098);
  // assert(nominal[12] == 7.977464881672084816e-05);

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
