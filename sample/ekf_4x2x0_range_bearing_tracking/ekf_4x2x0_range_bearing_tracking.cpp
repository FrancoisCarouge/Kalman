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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numbers>

namespace fcarouge::sample {
namespace {
template <auto Size> using vector = column_vector<double, Size>;
template <auto Row, auto Column> using matrix = matrix<double, Row, Column>;

//! @brief The angle in [-π, π).
const auto wrap{[](double angle) -> double {
  return std::remainder(angle, 2. * std::numbers::pi);
}};

//! @brief The predicted bearing [rad] and range [m] of a target at the
//! position of the state, from a sensor at the origin.
const auto predict_observation{[](const vector<4> &x) -> vector<2> {
  return vector<2>{std::atan2(x(1), x(0)), std::hypot(x(0), x(1))};
}};

//! @brief Tracking a target by bearing and range.
//!
//! @see Y. Bar-Shalom, X. R. Li, T. Kirubarajan, "Estimation with
//! Applications to Tracking and Navigation", Wiley, 2001: the constant velocity
//! model and the polar measurements.
//! @see
//! https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_cpp/bayes_tracking_example
//!
//! @details A sensor at the origin measures the bearing [rad] and the range
//! [m] of a target circling it at 8 m and 0.2 rad.s^-1, every 0.1 s, with a
//! bearing noise of 0.26 rad (15 degrees) and a range noise of 0.3 m. The
//! filter estimates the target's position and velocity, x, y [m], vx, vy
//! [m.s^-1], with a constant velocity model whose process noise absorbs the
//! turn. The observation is nonlinear: the filter linearizes it at the
//! predicted state.
//!
//! The bearing is an angle: as the target crosses the negative x axis, the
//! measured bearing jumps from π to -π while the predicted one does not, and
//! the innovation z - h(x) would be 2π instead of a few degrees. The measured
//! bearing is therefore unwrapped around the predicted one before each update.
//! Without it, each crossing throws the estimate up to 17 m off the target,
//! until the range observations pull it back; with it, the estimate stays
//! within about 2 m of the target throughout.
//!
//! @todo Support an angle difference for the innovation, so that the filter
//! wraps the bearing itself rather than its caller.
//!
//! @example ekf_4x2x0_range_bearing_tracking.cpp
[[maybe_unused]] const auto sample{[] -> int {
  const double time_step{0.1};                 // s
  const double radius{8.};                     // m
  const double angular_rate{0.2};              // rad.s^-1
  const double bearing_deviation{0.26};        // rad
  const double range_deviation{0.3};           // m
  const double position_noise_deviation{0.03}; // m
  const double velocity_noise_deviation{0.2};  // m.s^-1

  // The target starts on the negative x axis, at a bearing of nearly π.
  double phase{3.};
  // A deterministic standard normal sequence, for repeatable sensor noise: the
  // sum of twelve uniform deviates of a linear congruential generator, minus
  // six.
  auto noise{[seed = std::uint32_t{1}] mutable -> double {
    double sum{0.};
    for (int i{0}; i < 12; ++i) {
      seed = (seed * 1664525U) + 1013904223U;
      sum += static_cast<double>(seed) / 4294967296.;
    }
    return sum - 6.;
  }};

  kalman filter{
      // The state X: the target's position [m] and velocity [m.s^-1], from
      // its first observation, at rest.
      state{vector<4>{radius * std::cos(phase), radius * std::sin(phase), 0.,
                      0.}},
      // The output Z: the bearing [rad] and the range [m].
      output<vector<2>>,
      // The estimate uncertainty P [m^2, (m.s^-1)^2]:
      estimate_uncertainty{{25., 0., 0., 0.},
                           {0., 25., 0., 0.},
                           {0., 0., 1., 0.},
                           {0., 0., 0., 1.}},
      // The process uncertainty Q:
      process_uncertainty{
          {position_noise_deviation * position_noise_deviation, 0., 0., 0.},
          {0., position_noise_deviation * position_noise_deviation, 0., 0.},
          {0., 0., velocity_noise_deviation * velocity_noise_deviation, 0.},
          {0., 0., 0., velocity_noise_deviation * velocity_noise_deviation}},
      // The output uncertainty R [rad^2, m^2]:
      output_uncertainty{{bearing_deviation * bearing_deviation, 0.},
                         {0., range_deviation * range_deviation}},
      // The output model H = ∂h/∂X, at the state:
      output_model{[](const vector<4> &x) -> matrix<2, 4> {
        const double range_squared{(x(0) * x(0)) + (x(1) * x(1))};
        const double range{std::sqrt(range_squared)};
        return matrix<2, 4>{
            {-x(1) / range_squared, x(0) / range_squared, 0., 0.},
            {x(0) / range, x(1) / range, 0., 0.}};
      }},
      // The constant velocity model: its Jacobian F, the transition itself.
      state_transition{[time_step](const vector<4> &) -> matrix<4, 4> {
        return matrix<4, 4>{{1., 0., time_step, 0.},
                            {0., 1., 0., time_step},
                            {0., 0., 1., 0.},
                            {0., 0., 0., 1.}};
      }},
      transition{[time_step](const vector<4> &x) -> vector<4> {
        return vector<4>{x(0) + (time_step * x(2)), x(1) + (time_step * x(3)),
                         x(2), x(3)};
      }},
      // The observation h(X):
      observation{[](const vector<4> &x) -> vector<2> {
        return predict_observation(x);
      }},
      update_types<>, prediction_types<>};

  double maximum_error{0.};
  int crossings{0};
  double previous_bearing{wrap(phase)};
  for (int step{0}; step < 1000; ++step) {
    phase += angular_rate * time_step;
    const double x{radius * std::cos(phase)};
    const double y{radius * std::sin(phase)};
    const double measured_bearing{
        wrap(std::atan2(y, x) + (bearing_deviation * noise()))};
    const double measured_range{std::hypot(x, y) + (range_deviation * noise())};
    crossings += static_cast<int>(
        std::abs(measured_bearing - previous_bearing) > std::numbers::pi);
    previous_bearing = measured_bearing;

    filter.predict();

    // Unwrap the measured bearing around the predicted one, so that the
    // innovation is the angle between them.
    const double predicted_bearing{predict_observation(filter.x())(0)};
    filter.update(predicted_bearing +
                      wrap(measured_bearing - predicted_bearing),
                  measured_range);
    maximum_error = std::max(maximum_error,
                             std::hypot(filter.x()(0) - x, filter.x()(1) - y));
  }

  // The target crossed the negative x axis many times, and the estimate stayed
  // close to it throughout.
  assert(crossings >= 40);
  assert(maximum_error < 3. && "The estimated position stays near the target.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
