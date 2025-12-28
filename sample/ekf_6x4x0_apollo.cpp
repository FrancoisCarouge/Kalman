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
