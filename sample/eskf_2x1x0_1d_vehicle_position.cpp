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

namespace fcarouge::sample {
namespace {
template <auto Size> using vector = column_vector<double, Size>;
template <auto Row, auto Column> using matrix = matrix<double, Row, Column>;
using state = fcarouge::state<vector<2>>;

//! @brief Estimating a 1D vehicle position with an error-state filter.
//!
//! @details Estimate the position and velocity of a one-dimension vehicle
//! using an Error-State Kalman Filter (ESKF), also known as an indirect Kalman
//! filter. The nominal state, the best-guess position and velocity, is
//! integrated outside of the filter from a noisy inertial measurement unit
//! (IMU) accelerometer at 10Hz. The nominal state drifts as the accelerometer
//! noise accumulates. The filter does not estimate the position and velocity
//! but their errors: the difference between the true and nominal states. A
//! global navigation satellite system (GNSS) receiver measures the position at
//! the same rate. At each step, the error-state uncertainty is propagated, the
//! GNSS measurement corrects the error estimate, the estimated error is
//! injected into the nominal state, and the error state is reset to zero. The
//! error-state dynamics are linear and remain close to the zero origin, where
//! the linearization is the most accurate, even when the nominal dynamics are
//! nonlinear. The vehicle starts at 2 m.s^-1 and its true acceleration is
//! sin(t) m.s^-2. The sensor measurements are simulated over 10 seconds.
//!
//! @example eskf_2x1x0_1d_vehicle_position.cpp
[[maybe_unused]] const auto sample{[] -> int {
  // The constant 10Hz step period [s].
  const double dt{0.1};
  // The accelerometer noise variance [m2.s^-4].
  const double acceleration_variance{0.1};
  // The GNSS position noise variance [m2].
  const double position_variance{0.5};

  kalman filter{
      // The estimated error state X is the error in position [m] and in
      // velocity [m.s^-1]: [δp, δv]. The nominal state is initialized at the
      // best guess, the errors are thus initially zero.
      state{0., 0.},
      // The output Z is the observed position error [m]: the GNSS position
      // measurement minus the nominal position. The filter observes the error
      // state, not the position itself.
      output<double>,
      // The initial estimate uncertainty P of 1 m2 in position and 1 m2.s^-2 in
      // velocity.
      estimate_uncertainty{{1., 0.}, //
                           {0., 1.}},
      // The process uncertainty Q is the accelerometer noise integrated over
      // the step period, entering the velocity error and, through it, the
      // position error: Q = σa² G Gᵀ with G = [dt²/2, dt].
      process_uncertainty{{acceleration_variance * dt * dt * dt * dt / 4,
                           acceleration_variance * dt * dt * dt / 2},
                          {acceleration_variance * dt * dt * dt / 2,
                           acceleration_variance * dt * dt}},
      // The output uncertainty R is the GNSS noise variance.
      output_uncertainty{position_variance},
      // The output model H: the GNSS directly observes the position error.
      output_model{{1., 0.}},
      // The error-state transition F: the position error grows with the
      // velocity error over the step period.
      state_transition{{1., dt}, //
                       {0., 1.}}};

  // The nominal state, the best-guess estimate from the accelerometer
  // integration and the error injection, is tracked outside of the filter.
  double nominal_position{0.};
  double nominal_velocity{0.};

  // The simulated sensors measurements at each step: the accelerometer
  // acceleration [m.s^-2] and the GNSS position [m].
  struct measure {
    double acceleration;
    double position;
  };
  constexpr measure measured[]{{.acceleration = 0.054, .position = 0.078},
                               {.acceleration = 0.163, .position = 0.899},
                               {.acceleration = 0.255, .position = -0.452},
                               {.acceleration = 0.495, .position = 0.626},
                               {.acceleration = 0.411, .position = 1.109},
                               {.acceleration = 0.638, .position = 2.067},
                               {.acceleration = 0.852, .position = 1.546},
                               {.acceleration = 0.484, .position = 0.981},
                               {.acceleration = 0.861, .position = 2.863},
                               {.acceleration = 0.855, .position = 2.107},
                               {.acceleration = 1.059, .position = 1.409},
                               {.acceleration = 0.833, .position = 3.047},
                               {.acceleration = 1.24, .position = 2.803},
                               {.acceleration = 1.105, .position = 3.432},
                               {.acceleration = 1.245, .position = 2.762},
                               {.acceleration = 1.179, .position = 2.781},
                               {.acceleration = 0.163, .position = 3.736},
                               {.acceleration = 0.684, .position = 5.107},
                               {.acceleration = 1.156, .position = 3.958},
                               {.acceleration = 1.177, .position = 4.453},
                               {.acceleration = 0.836, .position = 5.304},
                               {.acceleration = 0.845, .position = 6.449},
                               {.acceleration = 0.948, .position = 6.484},
                               {.acceleration = 0.881, .position = 6.949},
                               {.acceleration = 0.4, .position = 6.483},
                               {.acceleration = 0.367, .position = 7.729},
                               {.acceleration = 0.348, .position = 9.418},
                               {.acceleration = 0.076, .position = 7.383},
                               {.acceleration = 0.482, .position = 9.563},
                               {.acceleration = 0.301, .position = 9.547},
                               {.acceleration = 0.493, .position = 9.289},
                               {.acceleration = -0.508, .position = 9.379},
                               {.acceleration = 0.144, .position = 9.133},
                               {.acceleration = -0.245, .position = 10.73},
                               {.acceleration = -0.451, .position = 11.456},
                               {.acceleration = -0.259, .position = 12.975},
                               {.acceleration = -0.334, .position = 11.287},
                               {.acceleration = -0.79, .position = 11.509},
                               {.acceleration = -0.387, .position = 12.069},
                               {.acceleration = -0.779, .position = 13.365},
                               {.acceleration = -1.047, .position = 12.985},
                               {.acceleration = -1.454, .position = 12.776},
                               {.acceleration = -1.096, .position = 14.175},
                               {.acceleration = -0.574, .position = 14.199},
                               {.acceleration = -0.895, .position = 14.651},
                               {.acceleration = -0.651, .position = 15.475},
                               {.acceleration = -0.913, .position = 14.43},
                               {.acceleration = -0.71, .position = 15.706},
                               {.acceleration = -0.594, .position = 15.696},
                               {.acceleration = -0.341, .position = 15.735},
                               {.acceleration = -0.422, .position = 16.332},
                               {.acceleration = -1.047, .position = 15.706},
                               {.acceleration = -0.88, .position = 17.755},
                               {.acceleration = -0.515, .position = 17.472},
                               {.acceleration = -1.457, .position = 17.717},
                               {.acceleration = -0.455, .position = 17.048},
                               {.acceleration = -0.749, .position = 17.652},
                               {.acceleration = 0.081, .position = 17.119},
                               {.acceleration = -0.509, .position = 19.035},
                               {.acceleration = -0.421, .position = 18.018},
                               {.acceleration = -0.151, .position = 17.6},
                               {.acceleration = -0.014, .position = 17.823},
                               {.acceleration = 0.297, .position = 18.88},
                               {.acceleration = 0.839, .position = 19.277},
                               {.acceleration = 0.647, .position = 18.359},
                               {.acceleration = 0.273, .position = 19.714},
                               {.acceleration = 0.957, .position = 18.506},
                               {.acceleration = 0.807, .position = 20.326},
                               {.acceleration = 1.063, .position = 20.63},
                               {.acceleration = 0.673, .position = 19.982},
                               {.acceleration = 0.334, .position = 20.72},
                               {.acceleration = 0.733, .position = 22.25},
                               {.acceleration = 0.657, .position = 21.295},
                               {.acceleration = 0.403, .position = 21.045},
                               {.acceleration = 1.021, .position = 22.173},
                               {.acceleration = 1.426, .position = 21.834},
                               {.acceleration = 0.635, .position = 22.473},
                               {.acceleration = 1.162, .position = 22.792},
                               {.acceleration = 0.777, .position = 23.55},
                               {.acceleration = 1.017, .position = 23.558},
                               {.acceleration = 1.373, .position = 23.818},
                               {.acceleration = 1.031, .position = 25.243},
                               {.acceleration = 0.979, .position = 23.855},
                               {.acceleration = 0.89, .position = 25.464},
                               {.acceleration = 0.836, .position = 25.143},
                               {.acceleration = 1.113, .position = 24.781},
                               {.acceleration = 0.117, .position = 25.73},
                               {.acceleration = 0.658, .position = 25.469},
                               {.acceleration = 0.776, .position = 26.715},
                               {.acceleration = 0.097, .position = 27.043},
                               {.acceleration = 0.257, .position = 26.022},
                               {.acceleration = 0.366, .position = 27.437},
                               {.acceleration = -0.111, .position = 28.237},
                               {.acceleration = 0.179, .position = 27.859},
                               {.acceleration = 0.051, .position = 29.385},
                               {.acceleration = -0.4, .position = 29.335},
                               {.acceleration = -0.271, .position = 31.064},
                               {.acceleration = -0.955, .position = 30.346},
                               {.acceleration = -0.554, .position = 30.172},
                               {.acceleration = 0.057, .position = 30.621}};

  for (const auto &[acceleration, position] : measured) {
    // The nominal state is propagated from the noisy accelerometer.
    nominal_position += (nominal_velocity * dt) + (acceleration * dt * dt / 2);
    nominal_velocity += acceleration * dt;

    // The error state is zero after each reset: the prediction propagates
    // the error-state uncertainty only.
    filter.predict();

    // The filter is updated with the observed position error.
    filter.update(position - nominal_position);

    // The estimated error is injected into the nominal state.
    nominal_position += filter.x()[0];
    nominal_velocity += filter.x()[1];

    // The error state is reset to zero since the error is now carried by the
    // nominal state. The covariance reset Jacobian is the identity for this
    // additive error, the estimate uncertainty is unchanged.
    filter.x(0., 0.);
  }

  // The nominal state, corrected by the filter, tracks the true state of
  // 30.63 m and 3.81 m.s^-1 within about three standard deviations of the
  // estimate uncertainty. Without the GNSS corrections, the integrated
  // accelerometer noise would drift away unbounded.
  assert(std::abs(nominal_position - 30.627) <
             3 * std::sqrt(filter.p()(0, 0)) &&
         std::abs(nominal_velocity - 3.810) < 3 * std::sqrt(filter.p()(1, 1)) &&
         "The nominal state expected within three standard deviations of the "
         "true state.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
