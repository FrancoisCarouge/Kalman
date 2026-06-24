/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.4
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
#include <random>

namespace fcarouge::sample {
namespace {
template <auto Size> using vector = column_vector<double, Size>;
template <auto Row, auto Column> using matrix = matrix<double, Row, Column>;
using state = fcarouge::state<vector<2>>;

//! @brief Estimating a 1D vehicle position.
//!
//! @details Estimate the position of a one-dimension vehicle using an
//! Error-State Kalman Filter (ESKF). The filter estimates the error state:
//! delta position and delta velocity. In this simplified example, the step
//! period is fixed at 10Hz. The filter models a constant velocity process with
//! the built-in period.
//!
//! @example eskf_2x1x0_1d_vehicle_position.cpp
[[maybe_unused]] auto sample{[] {
  // The best-guess, nominal state based on estimation and IMU integration is
  // externally tracked.
  double nominal_p{0.0};
  double nominal_v{0.0};

  // A built-in constant 10Hz step.
  const double dt{0.1};

  kalman filter{
      // The estimated error states X are the error in 1D position and in 1D
      // velocity: [delta_p, delta_v]. We start with no, zero error deltas: 0 m,
      // 0 m/s.
      state{0., 0.},

      // The measurement Z is the 1D GNSS position.
      // WHY WOULD THIS BE CALLED THE RESIDUAL?
      output<double>,

      // The initial estimate uncertainty P is a default uncertainty of 1 m2 for
      // position and 1 m2/s2 in velocity.
      estimate_uncertainty{{1., 0.}, //
                           {0., 1.}},

      // The process uncertainty Q is a small acceleration noise variance.
      process_uncertainty{{0., 0.}, //
                          {0., 0. * dt * dt}},

      // The output uncertainty R is the GPS sensor noise variance.
      output_uncertainty{0.5},

      // The output model H shows the direct observation of the position error
      // delta_p.
      output_model{1., 0.},

      // The state transition matrix F:
      state_transition{{1., dt}, //
                       {0., 1.}}

  };

  // 3. Simulation environment setup
  std::mt19937 generator{42};
  std::normal_distribution<double> noise_accel(0.0, std::sqrt(0.1));
  std::normal_distribution<double> noise_gps(0.0, std::sqrt(0.5));

  double true_p{0.0};
  double true_v{2.0}; // Vehicle moving at a constant true velocity

  std::println("Measured Acceleration, Measure GPS");

  // 4. Simulation loop
  for (int i = 0; i < 100; ++i) {
    // --- A. Simulate the "True" physical world ---
    double true_a = 0.0; // Assume constant velocity for truth
    double meas_a = true_a + noise_accel(generator); // Noisy IMU accelerometer

    true_p += true_v * dt + 0.5 * true_a * dt * dt;
    true_v += true_a * dt;

    // --- B. Predict Nominal State (IMU Integration) ---
    nominal_p += nominal_v * dt + 0.5 * meas_a * dt * dt;
    nominal_v += meas_a * dt;

    // --- C. Predict Error State Covariance ---
    // The actual error states are 0 right now, but we must propagate the
    // uncertainty
    filter.predict();

    // --- D. Update with GPS measurement ---
    double meas_gps = true_p + noise_gps(generator);

    std::println("{}, {}", meas_a, meas_gps);

    // The observation fed to the ESKF is the residual: z - nominal_p
    double residual = meas_gps - nominal_p;
    filter.update(residual);

    // --- E. Inject Error State into Nominal State ---
    auto error_state = filter.x();
    nominal_p += error_state(0);
    nominal_v += error_state(1);

    // --- F. Reset Error State ---
    // In an ESKF, the estimated error is transferred to the nominal state.
    // The error states must be explicitly reset to zero for the next iteration.
    filter.x(state{0.0, 0.0});
  }

  // 5. Verification
  // The estimated nominal position should converge to the true position.
  assert(std::abs(nominal_p - true_p) < 1.0 &&
         "ESKF estimation failed to converge.");

  std::println("Final estimated position: {}", nominal_p);
  std::println("Final estimated velocity: {}", nominal_v);

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
