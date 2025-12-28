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

#include "fcarouge/kalman.hpp"
#include "fcarouge/linalg.hpp"

#include <cassert>
#include <cmath>
#include <iomanip>

namespace fcarouge::sample {
namespace {
template <auto Size> using vector = column_vector<double, Size>;
template <auto Row, auto Column> using matrix = matrix<double, Row, Column>;
using state = fcarouge::state<vector<6>>;

//! @brief Apollo lunar module abort guidance system rendezvous filter.
//!
//! @copyright This example is transcribed from the NASA R-649/TN-D document.
//! The Apollo Rendezvous Navigation Filter Theory, Description and Performance,
//! Volume 1 of 2.
//!
//! @see https://archive.org/details/R649Volume1
//!
//! @details The 6x4 Extended Kalman Filter (EKF) Apollo Lunar Module (LM) Abort
//! Guidance System (AGS) for spacecraft rendezvous approaching the
//! Command/Service Module (CSM). The filter estimates the 3D position and
//! velocition of the spacecraft.
//! 6 States: Relative Position (rx, ry, rz), Relative Velocity (vx, vy, vz).
//! Predicts the state forward in time.
//!  The AGS simplified model assumes constant relative velocity between
//!  updates: assumption for the short time steps of the Apollo guidance computer).
// Measurement definition for the Rendezvous Radar.
//! 4 Measurements: Range (r), RangeRate (r_dot), Shaft (beta), Trunnion
//! (theta).
//!
//! @example ekf_6x4x0_apollo.cpp
[[maybe_unused]] auto apollo{[] {
  kalman filter{
      // The state X initialization scenario: The Lunar Module is 30km away,
      // approaching at 100 m/s relative to Command/Service Module.
      // Position: x=30km, y=1km, z=0.5km.
      // Velocity: Closing speed 100m/s.
      state{30000., 1000., 500., -100., 0., 0.},
      // The measurement definition Z for the Rendezvous Radar: Range (r),
      // RangeRate (r_dot), Shaft (beta), Trunnion (theta).
      output<vector<4>>,
      // The estimate uncertainty P is a high initial uncertainty in position
      // before radar lock. The velocity is somewhat known.
      estimate_uncertainty{{1000., 0., 0., 0., 0., 0.},
                           {0., 1000., 0., 0., 0., 0.},
                           {0., 0., 1000., 0., 0., 0.},
                           {0., 0., 0., 100., 0., 0.},
                           {0., 0., 0., 0., 100., 0.},
                           {0., 0., 0., 0., 0., 100.}},
      // The process uncertainty Q is a small accelerometer noise/drift.
      process_uncertainty{{0.01, 0., 0., 0., 0., 0.},
                          {0., 0.01, 0., 0., 0., 0.},
                          {0., 0., 0.01, 0., 0., 0.},
                          {0., 0., 0., 0.01, 0., 0.},
                          {0., 0., 0., 0., 0.01, 0.},
                          {0., 0., 0., 0., 0., 0.01}},
      // The output uncertainty R is the radar specifications.
      // Range error: ~30m, Rate error: ~0.5m/s, Angles error: ~0.005 rad.
      output_uncertainty{{30. * 30., 0., 0., 0.},
                         {0., 0.5 * 0.5, 0., 0.},
                         {0., 0., 0.005 * 0.005, 0.},
                         {0., 0., 0., 0.005 * 0.005}},
      // The output model H is the Jacobian, linearization around current
      // state. Complex derivation omitted for brevity, using a simplified
      // approximation.
      output_model{[](const vector<6> &x) -> matrix<4, 6> {
        // CAN THIS BE BINDED? SIMPLER SYNTAX?
        const auto rx{x[0]};
        const auto ry{x[1]};
        const auto rz{x[2]};
        const auto vx{x[3]};
        const auto vy{x[4]};
        const auto vz{x[5]};

        // 1. Calculate Predicted Measurement h(x)
        // Range
        double range = std::sqrt(rx * rx + ry * ry + rz * rz);
        //  Range Rate (Doppler)
        double range_rate = (rx * vx + ry * vy + rz * vz) / range;
        matrix<4, 6> h{matrix<4, 6>::Zero()};
        // d(Range) / d(State)
        h(0, 0) = rx / range;
        h(0, 1) = ry / range;
        h(0, 2) = rz / range;

        // d(RangeRate) / d(State): Complex derivation omitted for brevity,
        // simplified approximation.
        h(1, 0) = (vx * range - rx * range_rate) / (range * range); // dRR/dx
        h(1, 1) = (vy * range - ry * range_rate) / (range * range); // dRR/dy
        h(1, 2) = (vz * range - rz * range_rate) / (range * range); // dRR/dz
        h(1, 3) = rx / range;                                       // dRR/dvx
        h(1, 4) = ry / range;                                       // dRR/dvy
        h(1, 5) = rz / range;                                       // dRR/dvz

        // d(Shaft) / d(State)
        const auto r2_xy{rx * rx + ry * ry};
        h(2, 0) = -ry / r2_xy;
        h(2, 1) = rx / r2_xy;

        // d(Trunnion) / d(State)
        const auto term{std::sqrt(1 - (rz * rz) / (range * range))};
        h(3, 0) = (-rz * rx) / (range * range * range * term);
        h(3, 1) = (-rz * ry) / (range * range * range * term);
        h(3, 2) = (range * range - rz * rz) / (range * range * range * term);

        return h;
      }},
      // The state transition matrix F:
      // [ I  dt*I ]
      // [ 0   I   ]
      state_transition{[]([[maybe_unused]] const vector<6> &x,
                          const double &dt) -> matrix<6, 6> {
        matrix<6, 6> f{{1., 0., 0., dt, 0., 0.}, //
                       {0., 1., 0., 0., dt, 0.}, //
                       {0., 0., 1., 0., 0., dt}, //
                       {0., 0., 0., 1., 0., 0.}, //
                       {0., 0., 0., 0., 1., 0.}, //
                       {0., 0., 0., 0., 0., 1.}};
        return f;
      }},

      // Observation:
      observation{[](const vector<6> &x) {
        // CAN THIS BE BINDED? SIMPLER SYNTAX?
        const auto rx{x[0]};
        const auto ry{x[1]};
        const auto rz{x[2]};
        const auto vx{x[3]};
        const auto vy{x[4]};
        const auto vz{x[5]};

        // 1. Calculate Predicted Measurement h(x)
        // Range
        double range = std::sqrt(rx * rx + ry * ry + rz * rz);
        // Range Rate (Doppler)
        double range_rate = (rx * vx + ry * vy + rz * vz) / range;
        // Shaft Angle (Angle in XY plane usually, or defined by radar gimbal)
        // Simplified here as atan2(y, x) for illustration
        double shaft = std::atan2(ry, rx);
        // Trunnion Angle (Elevation)
        double trunnion = std::asin(rz / range);

        vector<4> z{range, range_rate, shaft, trunnion};
        return z;
      }},
      // The additional parameter for prediction: time update dt.
      prediction_types<double>};

  // Wrap angles if necessary (simple normalization) ??

  const double range{vector<3>{filter.x().template head<3>()}.norm()};
  std::println("Initial range: {} m", range);

  // --- Simulation Loop (10 seconds) ---
  for (int k = 0; k < 10; ++k) {
    const double dt{1.0}; // 1 second update loop
    // std::println("Before predict: {}", filter);
    filter.predict(dt);
    // std::println("After  predict: {}", filter);

    // 2. Simulate Receiving a Radar Measurement
    // (In a real system, this comes from hardware. Here we synthetically
    // generate it) True state would be approx: pos + vel*dt We observe:
    // Range=29900 (getting closer), Rate=-100, Angles~0
    // Fake data decreasing range by 100m each second
    const double true_range = 30000.0 - (100. * (k + 1));
    // Slight angle offsets
    filter.update(true_range, -100., 0.03, 0.01);
    // std::println("After  update: {}", filter);
    // NB: We may want to normalize, wrap the angles: modulo in [-Pi; +Pi].
    
    double r{vector<3>{filter.x().template head<3>()}.norm()};
    double v{filter.x()[3]};
    std::println("T+: {} s | Est Range: {} m | Est Closing Vel: {} m/s", k + 1,
                 r, v);
  }

  double r{vector<3>{filter.x().template head<3>()}.norm()};
  double v{filter.x()[3]};

  assert(std::abs(1 - r / 29'001.861'093'990'4) < 0.000'000'001 &&
         std::abs(1 - v / -99.574'527'631'012'3) < 0.000'000'001 &&
         "After simulating 10 seconds, the estimated range is 29,001.86 meters "
         "and an estimated closing velocity of 99.57 m/s.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample

///////////////////////////////////////////////////////////////////////////////
// As generated by Google's Gemini AI.
// TODO: Convert the style to declarative and conform to sample style.

//! @file ApolloRendezvousFilter.hpp
//! @brief A Modern C++20 implementation of the Apollo LM AGS Rendezvous Filter.
//! * Logic derived from Apollo Guidance System Operations (R-649/TN-D-6741):
//! - Dynamics: Linearized relative motion (assumed constant relative velocity
//! for AGS).
//! - Measurements: Rendezvous Radar (Range, Range-Rate, Shaft Angle, Trunnion
//! Angle).
//! - Style: FrancoisCarouge/Kalman (Heavily templated, strong types, Eigen
//! backend).

#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <numbers>

namespace apollo {

// -----------------------------------------------------------------------------
// 1. Strong Types & Concepts (Modern Style)
// -----------------------------------------------------------------------------

// Concept to ensure we only filter valid numerical types
template <typename T>
concept Numeric = std::floating_point<T>;

//! @brief State definition for the Rendezvous Filter.
//! Representing the Lunar Module (LM) relative to the Command Module (CSM).
//! 6 States: Relative Position (rx, ry, rz), Relative Velocity (vx, vy, vz).
template <Numeric T> struct State {
  Eigen::Matrix<T, 6, 1> vector;
  Eigen::Matrix<T, 6, 6> covariance;

  // Semantic Accessors
  [[nodiscard]] auto position() const { return vector.template head<3>(); }
  [[nodiscard]] auto velocity() const { return vector.template tail<3>(); }
};

//! @brief Measurement definition for the Rendezvous Radar.
//! 4 Measurements: Range (r), RangeRate (r_dot), Shaft (beta), Trunnion
//! (theta).
template <Numeric T> struct RadarMeasurement {
  static constexpr size_t Dim = 4;
  Eigen::Matrix<T, Dim, 1> value; // [r, r_dot, beta, theta]
  Eigen::Matrix<T, Dim, Dim> noise;
};

// -----------------------------------------------------------------------------
// 2. The Filter Implementation
// -----------------------------------------------------------------------------

//! @brief The Apollo AGS (Abort Guidance System) Filter.
//! Implements an Extended Kalman Filter (EKF) for spacecraft rendezvous.
template <Numeric T = double> class RendezvousFilter {
public:
  using StateType = State<T>;
  using MeasType = RadarMeasurement<T>;
  using Mat6 = Eigen::Matrix<T, 6, 6>;
  using Vec6 = Eigen::Matrix<T, 6, 1>;
  using Mat46 = Eigen::Matrix<T, 4, 6>;
  using Vec4 = Eigen::Matrix<T, 4, 1>;

  constexpr explicit RendezvousFilter(const StateType &initial_state)
      : state_(initial_state) {}

  //! @brief Predicts the state forward in time.
  //!  The AGS simplified model assumes constant relative velocity between
  //!  updates
  //! (valid for the short time steps of the Apollo guidance computer).
  //! * x_{k+1} = x_k + v_k * dt
  //! v_{k+1} = v_k
  void predict(T dt, const Mat6 &process_noise_Q) {
    // State Transition Matrix (F)
    // [ I  dt*I ]
    // [ 0   I   ]
    Mat6 F = Mat6::Identity();
    F.template block<3, 3>(0, 3) = Eigen::Matrix<T, 3, 3>::Identity() * dt;

    // 1. Project State
    state_.vector = F * state_.vector;

    // 2. Project Covariance
    state_.covariance = F * state_.covariance * F.transpose() + process_noise_Q;
  }

  //! @brief Updates the state with new Radar measurements.
  //! Uses EKF non-linear measurement equations.
  void update(const MeasType &measurement) {
    // Unpack current best estimate
    const auto rx = state_.vector(0);
    const auto ry = state_.vector(1);
    const auto rz = state_.vector(2);
    const auto vx = state_.vector(3);
    const auto vy = state_.vector(4);
    const auto vz = state_.vector(5);

    // 1. Calculate Predicted Measurement h(x)
    // Range
    T range = std::sqrt(rx * rx + ry * ry + rz * rz);
    // Range Rate (Doppler)
    T range_rate = (rx * vx + ry * vy + rz * vz) / range;
    // Shaft Angle (Angle in XY plane usually, or defined by radar gimbal)
    // Simplified here as atan2(y, x) for illustration
    T shaft = std::atan2(ry, rx);
    // Trunnion Angle (Elevation)
    T trunnion = std::asin(rz / range);

    Vec4 z_pred;
    z_pred << range, range_rate, shaft, trunnion;

    // 2. Calculate Jacobian H (Linearization around current state)
    Mat46 H = Mat46::Zero();

    // -- d(Range) / d(State) --
    H(0, 0) = rx / range;
    H(0, 1) = ry / range;
    H(0, 2) = rz / range;

    // -- d(RangeRate) / d(State) --
    // (Complex derivation omitted for brevity, simplified approximation)
    H(1, 0) = (vx * range - rx * range_rate) / (range * range); // dRR/dx
    H(1, 1) = (vy * range - ry * range_rate) / (range * range); // dRR/dy
    H(1, 2) = (vz * range - rz * range_rate) / (range * range); // dRR/dz
    H(1, 3) = rx / range;                                       // dRR/dvx
    H(1, 4) = ry / range;                                       // dRR/dvy
    H(1, 5) = rz / range;                                       // dRR/dvz

    // -- d(Shaft) / d(State) --
    T r2_xy = rx * rx + ry * ry;
    H(2, 0) = -ry / r2_xy;
    H(2, 1) = rx / r2_xy;

    // -- d(Trunnion) / d(State) --
    T term = std::sqrt(1 - (rz * rz) / (range * range));
    H(3, 0) = (-rz * rx) / (range * range * range * term);
    H(3, 1) = (-rz * ry) / (range * range * range * term);
    H(3, 2) = (range * range - rz * rz) / (range * range * range * term);

    // 3. EKF Standard Update
    // Innovation
    Vec4 y = measurement.value - z_pred;

    // Wrap angles if necessary (simple normalization)
    while (y(2) > std::numbers::pi_v<T>)
      y(2) -= 2 * std::numbers::pi_v<T>;
    while (y(2) < -std::numbers::pi_v<T>)
      y(2) += 2 * std::numbers::pi_v<T>;

    // S = H P H^t + R
    Eigen::Matrix<T, 4, 4> S =
        H * state_.covariance * H.transpose() + measurement.noise;

    // K = P H^t S^-1
    Eigen::Matrix<T, 6, 4> K = state_.covariance * H.transpose() * S.inverse();

    // x = x + K y
    state_.vector = state_.vector + K * y;

    // P = (I - KH) P
    state_.covariance = (Mat6::Identity() - K * H) * state_.covariance;
  }

  [[nodiscard]] const StateType &state() const { return state_; }

private:
  StateType state_;
};

} // namespace apollo

// -----------------------------------------------------------------------------
// 3. Example Usage (Simulating LM Approach)
// -----------------------------------------------------------------------------

int main() {
  using T = double;

  // --- Initialization ---
  // Scenario: LM is 30km away, approaching at 100 m/s relative to CSM
  apollo::State<T> initial_state;
  initial_state.vector << 30000.0, 1000.0, 500.0, // Pos: x=30km, y=1km, z=0.5km
      -100.0, 0.0, 0.0;                           // Vel: Closing speed 100m/s

  // Initial uncertainty (High uncertainty in position before radar lock)
  initial_state.covariance = Eigen::Matrix<T, 6, 6>::Identity() * 1000.0;
  initial_state.covariance.block<3, 3>(3, 3) *= 0.1; // Velocity somewhat known

  apollo::RendezvousFilter<T> ags_filter(initial_state);

  // --- Matrices Configuration ---
  // Process Noise (Q): Small accelerometer noise / drift
  Eigen::Matrix<T, 6, 6> Q = Eigen::Matrix<T, 6, 6>::Identity() * 0.01;

  // Measurement Noise (R): Radar specifications
  // Range error ~30m, Rate error ~0.5m/s, Angle error ~0.005 rad
  apollo::RadarMeasurement<T> meas_config;
  meas_config.noise = Eigen::Matrix<T, 4, 4>::Identity();
  meas_config.noise.diagonal() << 30.0 * 30.0, 0.5 * 0.5, 0.005 * 0.005,
      0.005 * 0.005;

  std::cout << "--- Apollo AGS Rendezvous Filter Simulation ---\n";
  std::cout << "Initial Range: " << initial_state.position().norm() << " m\n\n";

  // --- Simulation Loop (10 seconds) ---
  T dt = 1.0; // 1 second update loop
  for (int k = 0; k < 10; ++k) {

    // std::cout << "Before predict: " << ags_filter.state().vector;
    // 1. Predict
    ags_filter.predict(dt, Q);
    // std::cout << "After  predict: " << ags_filter.state().vector;

    // 2. Simulate Receiving a Radar Measurement
    // (In a real system, this comes from hardware. Here we synthetically
    // generate it) True state would be approx: pos + vel*dt We observe:
    // Range=29900 (getting closer), Rate=-100, Angles~0

    apollo::RadarMeasurement<T> z;
    z.noise = meas_config.noise;

    // Fake data decreasing range by 100m each second
    T true_range = 30000.0 - (100.0 * (k + 1));
    z.value << true_range, -100.0, 0.03, 0.01; // Slight angle offsets

    // 3. Update
    ags_filter.update(z);
    // std::cout << "After  update: " << ags_filter.state().vector;

    // Output
    auto current_state = ags_filter.state();
    std::cout << std::setprecision(15) << "T+" << (k + 1) << "s | "
              << "Est Range: " << current_state.position().norm() << " m | "
              << "Est Closing Vel: " << current_state.velocity().x()
              << " m/s\n";
  }

  return 0;
}
