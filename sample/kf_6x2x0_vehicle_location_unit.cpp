/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.4.0
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
template <auto... References>
using vector = fcarouge::vector<double, References...>;
using state = fcarouge::state<vector<m, m / s, m / s2, m, m / s, m / s2>>;
using output_t = vector<m, m>;
using estimate_uncertainty =
    fcarouge::estimate_uncertainty<ᴀʙᵀ<state::type, state::type>>;
using process_uncertainty =
    fcarouge::process_uncertainty<ᴀʙᵀ<state::type, state::type>>;
using output_uncertainty =
    fcarouge::output_uncertainty<ᴀʙᵀ<output_t, output_t>>;
using output_model = fcarouge::output_model<ᴀʙᵀ<output_t, state::type>>;
using state_transition =
    fcarouge::state_transition<ᴀʙᵀ<state::type, state::type>>;

template <auto... Reference>
inline fcarouge::output_t<vector<Reference...>> output{
    fcarouge::output<vector<Reference...>>};

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
      state{0. * m, 0. * m / s, 0. * m / s2, 0. * m, 0. * m / s, 0. * m / s2},
      // The vehicle has an onboard location sensor that reports output Z as X
      // and Y coordinates of the system.
      output<m, m>,
      // The estimate uncertainty matrix P.
      // Since our initial state vector is a guess, we will set a very high
      // estimate uncertainty. The high estimate uncertainty results in a high
      // Kalman Gain, giving a high weight to the measurement.
      estimate_uncertainty{500. * one<estimate_uncertainty::type>},
      // The process uncertainty noise matrix Q, constant, computed in place,
      // with  random acceleration standard deviation: σa = 0.2 m.s^-2.
      process_uncertainty{[]() {
        process_uncertainty::type value{one<process_uncertainty::type>};
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
        output_model::type value{zero<output_model::type>};
        value.at<0, 0>() = 1. * m2;
        value.at<1, 3>() = 1. * m2;
        return value;
      }()},
      // The state transition matrix F would be:
      state_transition{[]() {
        state_transition::type value{one<state_transition::type>};
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
  assert(abs(1 - filter.x().at<0>() / (-277.8 * m)) < 0.001 &&
         abs(1 - filter.x().at<1>() / (148.3 * m / s)) < 0.001 &&
         abs(1 - filter.x().at<2>() / (94.5 * m / s2)) < 0.001 &&
         abs(1 - filter.x().at<3>() / (249.8 * m)) < 0.001 &&
         abs(1 - filter.x().at<4>() / (-85.9 * m / s)) < 0.001 &&
         abs(1 - filter.x().at<5>() / (-63.62 * m / s2)) < 0.001 &&
         "The state estimates expected at 0.1% accuracy.");

  //! @todo Add format verification.
  //! @todo Complete the example from the original data.

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
