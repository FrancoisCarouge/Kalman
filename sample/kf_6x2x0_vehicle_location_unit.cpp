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
#include <format>

namespace fcarouge::test {
namespace {
template <typename... Types> using vector = column_vector<double, Types...>;
using state_t =
    vector<position, velocity, acceleration, position, velocity, acceleration>;
using output_t = vector<position, position>;

//! @brief Estimating the vehicle location.
//!
//! @copyright This example is transcribed from KalmanFilter.NET copyright Alex
//! Becker.
//!
//! @see https://www.kalmanfilter.net/multiExamples.html#ex9
//!
//! @details In this example, we would like to estimate the location of the
//! vehicle in the XY plane. The vehicle has an onboard location sensor that
//! reports X and Y coordinates of the system. We assume constant acceleration
//! dynamics. In this example we don't have a control variable u since we don't
//! have control input. Let us assume a vehicle moving in a straight line in the
//! X direction with a constant velocity. After traveling 400 meters the vehicle
//! turns right, with a turning radius of 300 meters. During the turning
//! maneuver, the vehicle experiences acceleration due to the circular motion
//! (an angular acceleration). The measurements period: Δt = 1s (constant).
//!
//! @example kf_6x2x0_vehicle_location_unit.cpp
[[maybe_unused]] auto sample{[] {
  // A 6x2x0 filter, constant acceleration dynamic model, no control.
  kalman filter{
      // The state X is chosen to be the position, velocity, acceleration in the
      // XY plane: [px, vx, ax, py, vy, ay]. We don't know the vehicle location;
      // we will set initial position, velocity and acceleration to 0.
      state{state_t{position{0. * m}, velocity{0. * m / s},
                    acceleration{0. * m / s2}, position{0. * m},
                    velocity{0. * m / s}, acceleration{0. * m / s2}}},
      // The vehicle has an onboard location sensor that reports output Z as X
      // and Y coordinates of the system.
      output<output_t>,
      // The estimate uncertainty matrix P.
      // Since our initial state vector is a guess, we will set a very high
      // estimate uncertainty. The high estimate uncertainty results in a high
      // Kalman Gain, giving a high weight to the measurement.
      estimate_uncertainty{
          500. * kalman_internal::one<kalman_internal::ᴀʙᵀ<state_t, state_t>>},
      // The process uncertainty noise matrix Q, constant, computed in place,
      // with  random acceleration standard deviation: σa = 0.2 m.s^-2.
      process_uncertainty{[]() {
        using process_uncertainty_t = kalman_internal::ᴀʙᵀ<state_t, state_t>;
        process_uncertainty_t value{
            kalman_internal::one<process_uncertainty_t>};
        value.at<0, 0>() = 0.25 * m2;
        value.at<0, 1>() = 0.5 * m2 / s;
        value.at<0, 2>() = 0.5 * m2 / s2;
        value.at<1, 0>() = 0.5 * m2 / s;
        value.at<1, 2>() = 1. * m2 / s3;
        value.at<2, 0>() = 0.5 * m2 / s2;
        value.at<2, 1>() = 1. * m2 / s3;
        value.at<3, 3>() = 0.25 * m2;
        value.at<3, 4>() = 0.5 * m2 / s;
        value.at<3, 5>() = 0.5 * m2 / s2;
        value.at<4, 3>() = 0.5 * m2 / s;
        value.at<4, 5>() = 1. * m2 / s3;
        value.at<5, 3>() = 0.5 * m2 / s2;
        value.at<5, 4>() = 1. * m2 / s3;
        return 0.2 * 0.2 * value;
      }()},
      // The output uncertainty matrix R. Assume that the x and y measurements
      // are uncorrelated, i.e. error in the x coordinate measurement doesn't
      // depend on the error in the y coordinate measurement. In real-life
      // applications, the measurement uncertainty can differ between
      // measurements. In many systems the measurement uncertainty depends on
      // the measurement SNR (signal-to-noise ratio), angle between sensor (or
      // sensors) and target, signal frequency and many other parameters.
      // For the sake of the example simplicity, we will assume a constant
      // measurement uncertainty: R1 = R2...Rn-1 = Rn = R The measurement error
      // standard deviation: σxm = σym = 3m. The variance 9.
      output_uncertainty{{9. * m2, 0. * m2}, {0. * m2, 9. * m2}},
      // The output model matrix H. The dimension of zn is 2x1 and the dimension
      // of xn is 6x1. Therefore the dimension of the observation matrix H shall
      // be 2x6.
      output_model{[]() {
        using output_model_t = kalman_internal::evaluate<
            kalman_internal::quotient<output_t, state_t>>;
        output_model_t value{kalman_internal::zero<output_model_t>};
        value.at<0, 0>() = 1. * m2;
        value.at<1, 3>() = 1. * m2;
        return value;
      }()},
      // The state transition matrix F would be:
      state_transition{[]() {
        using state_transition_t = kalman_internal::evaluate<
            kalman_internal::quotient<state_t, state_t>>;
        state_transition_t value{kalman_internal::one<state_transition_t>};
        value.at<0, 1>() = 1. * m2 / s;
        value.at<0, 2>() = 0.5 * m2 / s2;
        value.at<1, 2>() = 1. * m2 / s3;
        value.at<3, 4>() = 1. * m2 / s;
        value.at<3, 5>() = 0.5 * m2 / s2;
        value.at<4, 5>() = 1. * m2 / s3;
        return value;
      }()}};

  // Now we can predict the next state based on the initialization values.
  filter.predict();

  // The measurement values: z1 = [-393.66 m, 300.4 m]
  filter.update(-393.66 * m, 300.4 * m);
  filter.predict();
  filter.update(-375.93 * m, 301.78 * m);
  filter.predict();

  // Verify the example estimated state at 0.1% accuracy.
  assert(position{-277.8 * m} < filter.x().at<0>() &&
         filter.x().at<0>() < position{-277.7 * m} &&
         velocity{148.3 * m / s} < filter.x().at<1>() &&
         filter.x().at<1>() < velocity{148.4 * m / s} &&
         acceleration{94.5 * m / s2} < filter.x().at<2>() &&
         filter.x().at<2>() < acceleration{94.56 * m / s2} &&
         position{249.7 * m} < filter.x().at<3>() &&
         filter.x().at<3>() < position{249.8 * m} &&
         velocity{-86.0 * m / s} < filter.x().at<4>() &&
         filter.x().at<4>() < velocity{-85.9 * m / s} &&
         acceleration{-63.65 * m / s2} < filter.x().at<5>() &&
         filter.x().at<5>() < acceleration{-63.64 * m / s2} &&
         "The state estimates expected at 0.1% accuracy.");

  //! @todo Add format verification.
  //! @todo Complete the example from the original data.

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
