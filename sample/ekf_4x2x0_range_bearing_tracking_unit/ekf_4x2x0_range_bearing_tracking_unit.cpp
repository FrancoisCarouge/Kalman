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
#include "fcarouge/realtime.hpp"

#include <cassert>
#include <numbers>
#include <random>
#include <tuple>

#include <mp-units/math.h>
#include <mp-units/systems/isq/space_and_time.h>
#include <mp-units/systems/si.h>
#include <mp-units/systems/si/math.h>

namespace fcarouge::sample {
namespace {
//! @name The Axes of the Sensor Frame
//! @brief Each axis is a quantity kind of its own: an x position is never added
//! to, nor substituted for, a y position. A velocity is the position of its
//! axis by a duration, so that a velocity over a time step is a position of the
//! same axis.
//! @{
constexpr struct x_position
    : mp_units::quantity_spec<mp_units::isq::displacement, mp_units::is_kind> {
} x_position;
constexpr struct y_position
    : mp_units::quantity_spec<mp_units::isq::displacement, mp_units::is_kind> {
} y_position;
constexpr struct x_velocity
    : mp_units::quantity_spec<x_position / mp_units::isq::duration> {
} x_velocity;
constexpr struct y_velocity
    : mp_units::quantity_spec<y_position / mp_units::isq::duration> {
} y_velocity;
//! @}

//! @name The Polar Coordinates of the Target
//! @brief The bearing, the angle from the x axis toward the y axis, and the
//! range, the distance from the sensor.
//! @{
constexpr struct bearing
    : mp_units::quantity_spec<mp_units::isq::angular_measure,
                              mp_units::is_kind> {
} bearing;
constexpr struct range
    : mp_units::quantity_spec<mp_units::isq::distance, mp_units::is_kind> {
} range;
//! @}

using mp_units::quantity;
using mp_units::si::unit_symbols::m;
using mp_units::si::unit_symbols::m2;
using mp_units::si::unit_symbols::rad;
using mp_units::si::unit_symbols::s;
using mp_units::si::unit_symbols::s2;

//! @brief A position along the x axis [m].
using abscissa = quantity<x_position[m]>;
//! @brief A position along the y axis [m].
using ordinate = quantity<y_position[m]>;
//! @brief A velocity along the x axis [m.s^-1].
using abscissa_rate = quantity<x_velocity[m / s]>;
//! @brief A velocity along the y axis [m.s^-1].
using ordinate_rate = quantity<y_velocity[m / s]>;
//! @brief The angle of the target from the x axis toward the y axis [rad].
using azimuth = quantity<bearing[rad]>;
//! @brief The distance of the target from the sensor [m].
using distance = quantity<range[m]>;
//! @brief A length, the quantity common to both axes [m].
using length = quantity<mp_units::isq::length[m]>;
//! @brief A squared length [m^2].
using squared_length = quantity<pow<2>(mp_units::isq::length)[m2]>;

template <typename... Types> using vector = column_vector<double, Types...>;
//! @brief The track, the state X: the target's position [m] and velocity
//! [m.s^-1].
using track = vector<abscissa, ordinate, abscissa_rate, ordinate_rate>;
//! @brief The measurement, the output Z: the target's bearing [rad] and range
//! [m].
using measurement = vector<azimuth, distance>;
//! @brief The track elements: the row and column indexes of a covariance of
//! the track.
using track_indexes = track::row_indexes;
//! @brief The measurement elements: the row and column indexes of a
//! covariance of the measurement.
using measurement_indexes = measurement::row_indexes;
//! @brief The reciprocals of the track elements: the column indexes of a
//! derivative by the track.
using reciprocal_track =
    std::tuple<quantity<mp_units::one / x_position[m]>,
               quantity<mp_units::one / y_position[m]>,
               quantity<mp_units::one / x_velocity[m / s]>,
               quantity<mp_units::one / y_velocity[m / s]>>;
//! @brief The derivatives of the track by the track, the state transition F.
//! The element of a row and a column is the product of the track element of
//! the row by the reciprocal of the track element of the column.
using track_jacobian = matrix<double, track_indexes, reciprocal_track>;
//! @brief The derivatives of the measurement by the track, the output model H.
//! The element of a row and a column is the product of the measurement element
//! of the row by the reciprocal of the track element of the column.
using measurement_jacobian =
    matrix<double, measurement_indexes, reciprocal_track>;
//! @brief A covariance of the track, such as the estimate uncertainty P: the
//! element of a row and a column is the product of the track elements they
//! index.
using track_covariance = matrix<double, track_indexes, track_indexes>;
//! @brief A covariance of the measurement, the output uncertainty R: the
//! element of a row and a column is the product of the measurement elements
//! they index.
using measurement_covariance =
    matrix<double, measurement_indexes, measurement_indexes>;

//! @brief The period of the sensor.
constexpr auto time_step{0.1 * s};

//! @brief A bearing wrapped within a half turn: [-π, π] rad.
const auto wrap{[](const azimuth &angle) -> azimuth {
  return remainder(angle, 2. * std::numbers::pi * bearing[rad]);
}};

//! @brief The bearing and range of a position, from the sensor at the origin.
//!
//! @details The polar coordinates combine both axes, each a quantity kind of
//! its own: each position is restated as a length, their common quantity,
//! whose angle is the bearing and whose hypotenuse is restated as the range.
const auto polar{
    [](const abscissa &position_x, const ordinate &position_y) -> measurement {
      const length along_x{position_x};
      const length along_y{position_y};
      return measurement{mp_units::si::atan2(along_y, along_x),
                         distance{hypot(along_x, along_y)}};
    }};

//! @brief The output model H = ∂h/∂X of the polar coordinates, at a state.
//!
//! @details The bearing derives from the x position by -y/r², from the y
//! position by x/r², the quantities bearing per x or y position. The range
//! derives from them by the components of the unit vector toward the target,
//! x/r and y/r. Neither depends on the velocity. An angle is an arc length by
//! its radius: the ratios of the lengths are restated as the derivatives.
const auto output_model_of{[](const track &estimate) -> measurement_jacobian {
  const length along_x{estimate.at<0>()};
  const length along_y{estimate.at<1>()};
  const squared_length squared{(along_x * along_x) + (along_y * along_y)};
  const length hypotenuse{hypot(along_x, along_y)};
  measurement_jacobian value;
  value.at<0, 0>((bearing / x_position)(-along_y / squared));
  value.at<0, 1>((bearing / y_position)(along_x / squared));
  value.at<1, 0>((range / x_position)(along_x / hypotenuse));
  value.at<1, 1>((range / y_position)(along_y / hypotenuse));
  return value;
}};

//! @brief Tracking a target by bearing and range.
//!
//! @copyright This example is inspired by the Bayesian tracking example of the
//! Mobile Robot Programming Toolkit (MRPT), by the MRPT authors. The model is
//! written from its equations, not transcribed from its code.
//!
//! @see Yaakov Bar-Shalom, X. Rong Li, Thiagalingam Kirubarajan, Estimation
//! with Applications to Tracking and Navigation, John Wiley & Sons, 2001, the
//! constant velocity model, the polar measurements, and the normalized
//! innovation squared consistency test, https://doi.org/10.1002/0471221279
//! @see
//! https://github.com/MRPT/mrpt/tree/develop/mrpt_examples_cpp/bayes_tracking_example
//!
//! @details A sensor at the origin measures the bearing and the range of a
//! target circling it, ten times a second. The bearing is noisy, 0.26 rad (15
//! degrees), the range precise, 0.3 m: at 8 m, a single measurement places the
//! target within 2.1 m across the line of sight, but 0.3 m along it:
//!
//! @image html ekf_4x2x0_range_bearing_tracking_unit_layout.svg
//!
//! The state is the target's position and velocity, each axis a quantity kind
//! of its own. A constant velocity model predicts the target, its process noise
//! absorbing the turn. The observation is nonlinear: the polar coordinates of
//! the position. The output model H = ∂h/∂X is evaluated at the predicted
//! state.
//!
//! The bearing is an angle: as the target crosses the negative x axis, the
//! measured bearing jumps from π to -π while the predicted one does not, and
//! the innovation z - h(X) would be nearly 2π instead of a few degrees. The
//! measured bearing is therefore unwrapped around the predicted one before each
//! update. In the plotted run, fed the raw bearing, the same filter is thrown
//! up to 22.7 m off the target at each crossing, until the ranges pull it back.
//! With the unwrapped bearing, its error stays within 1.9 m throughout, 0.60 m
//! root mean square.
//!
//! The filter recovers the cross-range position the bearing lacks from the
//! motion of the target: its cross-range error, 0.58 m root mean square, is a
//! third of a single bearing's, while its down-range error is 0.15 m. Both are
//! within their estimated standard deviations, 0.73 m and 0.18 m, the square
//! roots of the estimate uncertainty P across and along the line of sight. The
//! normalized innovation squared, yᵀ S⁻¹ y, averages 1.93 for its two degrees
//! of freedom, and 95% of its values lie within the 95% chi-squared bound: the
//! filter is consistent. The raw bearing's innovations reach a normalized
//! square of 474 at the crossings.
//!
//! The plots compare the filter of this sample with the same filter fed the raw
//! bearing, over the same measurements:
//!
//! @image html ekf_4x2x0_range_bearing_tracking_unit.svg
//!
//! @todo Support an angle difference for the innovation, so that the filter
//! wraps the bearing itself rather than its caller.
//!
//! @example ekf_4x2x0_range_bearing_tracking_unit.cpp
[[maybe_unused]] const auto sample{[] -> int {
  const auto radius{8. * m};
  const auto angular_rate{0.2 * rad / s};
  const auto bearing_deviation{0.26 * rad};
  const auto range_deviation{0.3 * m};
  const auto position_noise{0.03 * m};
  const auto velocity_noise{0.2 * m / s};

  // The target starts near the negative x axis, at a bearing of nearly π.
  auto phase{3. * rad};
  // The seed of the sensor noise, from the system's entropy source: a
  // blocking input-output operation, out of the real-time verification.
  const auto seed{[] -> std::random_device::result_type {
    const not_realtime opt_out;
    return std::random_device{}();
  }()};
  // The sensor noise, a standard normal sequence. The plots are of the seed 1,
  // with the GNU C++ standard library: the algorithm of the distribution, so
  // the sequence of a seed, varies across standard libraries.
  std::minstd_rand engine{seed};
  std::normal_distribution distribution{0., 1.};
  const auto noise{[&] -> double { return distribution(engine); }};

  kalman filter{
      // The state X: the target's position, from its first observation, at
      // rest.
      state{track{abscissa{radius * mp_units::si::cos(phase)},
                  ordinate{radius * mp_units::si::sin(phase)},
                  abscissa_rate{0. * m / s}, ordinate_rate{0. * m / s}}},
      // The output Z: the bearing and the range.
      output<measurement>,
      // The estimate uncertainty P:
      estimate_uncertainty{[]() -> track_covariance {
        track_covariance value;
        value.at<0, 0>(25. * m2);
        value.at<1, 1>(25. * m2);
        value.at<2, 2>(1. * m2 / s2);
        value.at<3, 3>(1. * m2 / s2);
        return value;
      }()},
      // The process uncertainty Q, whose velocity noise absorbs the turn:
      process_uncertainty{[&]() -> track_covariance {
        track_covariance value;
        value.at<0, 0>(pow<2>(position_noise));
        value.at<1, 1>(pow<2>(position_noise));
        value.at<2, 2>(pow<2>(velocity_noise));
        value.at<3, 3>(pow<2>(velocity_noise));
        return value;
      }()},
      // The output uncertainty R:
      output_uncertainty{[&]() -> measurement_covariance {
        measurement_covariance value;
        value.at<0, 0>(pow<2>(bearing_deviation));
        value.at<1, 1>(pow<2>(range_deviation));
        return value;
      }()},
      // The output model H = ∂h/∂X, at the state:
      output_model{output_model_of},
      // The constant velocity model: its Jacobian F, and the transition itself.
      state_transition{[](const track &) -> track_jacobian {
        track_jacobian value{track_jacobian::matrix::Identity()};
        value.at<0, 2>(time_step);
        value.at<1, 3>(time_step);
        return value;
      }},
      transition{[](const track &estimate) -> track {
        return track{estimate.at<0>() + (estimate.at<2>() * time_step),
                     estimate.at<1>() + (estimate.at<3>() * time_step),
                     estimate.at<2>(), estimate.at<3>()};
      }},
      // The observation h(X), the polar coordinates of the position:
      observation{[](const track &estimate) -> measurement {
        return polar(estimate.at<0>(), estimate.at<1>());
      }}};

  const int steps{1000};
  squared_length squared_errors{0. * m2};
  int crossings{0};
  azimuth previous_bearing{wrap(azimuth{phase})};
  for (int step{0}; step < steps; ++step) {
    phase += angular_rate * time_step;
    const abscissa target_x{radius * mp_units::si::cos(phase)};
    const ordinate target_y{radius * mp_units::si::sin(phase)};
    const measurement target{polar(target_x, target_y)};
    const azimuth measured_bearing{
        wrap(target.at<0>() + (bearing_deviation * noise()))};
    const distance measured_range{target.at<1>() + (range_deviation * noise())};
    crossings += static_cast<int>(abs(measured_bearing - previous_bearing) >
                                  std::numbers::pi * rad);
    previous_bearing = measured_bearing;

    filter.predict();

    // Unwrap the measured bearing around the predicted one, so that the
    // innovation is the angle between them.
    const azimuth predicted_bearing{
        polar(filter.x().at<0>(), filter.x().at<1>()).at<0>()};
    filter.update(predicted_bearing +
                      wrap(measured_bearing - predicted_bearing),
                  measured_range);

    squared_errors += pow<2>(length{filter.x().at<0>() - target_x}) +
                      pow<2>(length{filter.x().at<1>() - target_y});
  }

  // The target crossed the negative x axis four times, the measured bearing
  // jumping from π to -π at each crossing, and the estimate stayed close to the
  // target, whatever the seed of the noise.
  assert(crossings >= 4 &&
         "The measured bearing jumps at each crossing of the negative x axis.");
  assert(sqrt(squared_errors / steps) < 1. * m &&
         "The estimated position stays near the target.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
