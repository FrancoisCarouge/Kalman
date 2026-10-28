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

#include <array>
#include <cassert>
#include <cstddef>
#include <random>
#include <tuple>

#include <mp-units/math.h>
#include <mp-units/systems/isq/space_and_time.h>
#include <mp-units/systems/si.h>
#include <mp-units/systems/si/math.h>

namespace fcarouge::sample {
namespace {
using mp_units::quantity;
using mp_units::si::unit_symbols::m;
using mp_units::si::unit_symbols::m2;
using mp_units::si::unit_symbols::rad;
using mp_units::si::unit_symbols::s;
using mp_units::si::unit_symbols::s2;
using mp_units::si::unit_symbols::s3;
using position = quantity<mp_units::isq::length[m]>;
using velocity = quantity<mp_units::isq::velocity[m / s]>;
using acceleration = quantity<mp_units::isq::acceleration[m / s2]>;
//! @brief A squared position [m^2].
using squared_position = quantity<pow<2>(mp_units::isq::length)[m2]>;

template <typename... Types> using vector = column_vector<double, Types...>;
//! @brief The track, the state X: the target's position [m] and velocity
//! [m.s^-1] relative to the vehicle.
using track = vector<position, velocity>;
//! @brief The sighting, the output Z: the camera's measured position of the
//! target relative to the vehicle [m].
using sighting = vector<position>;
//! @brief The reciprocals of the track elements: the column indexes of a
//! derivative by the track.
using reciprocal_track =
    std::tuple<quantity<mp_units::one / mp_units::isq::length[m]>,
               quantity<mp_units::one / mp_units::isq::velocity[m / s]>>;
//! @brief A covariance of the track, the estimate uncertainty P and the process
//! uncertainty Q: the element of a row and a column is the product of the track
//! elements they index [m^2, m^2.s^-1, m^2.s^-2].
using track_covariance = matrix<double, track::row_indexes, track::row_indexes>;
//! @brief The state transition F: the derivatives of the track by the track
//! [1, s].
using track_transition = matrix<double, track::row_indexes, reciprocal_track>;
//! @brief The output model H: the derivatives of the sighting by the track [1,
//! s].
using sighting_model = matrix<double, sighting::row_indexes, reciprocal_track>;
//! @brief The input control G: the derivatives of the track by the vehicle's
//! acceleration [s^2, s].
using acceleration_control = matrix<
    double, track::row_indexes,
    std::tuple<quantity<mp_units::one / mp_units::isq::acceleration[m / s2]>>>;

//! @brief The period of the predictions, of the accelerometer.
constexpr auto time_step{0.02 * s};

//! @brief Tracking a target from delayed vision measurements.
//!
//! @copyright This example is inspired by the out-of-sequence measurement
//! handling of the vision target estimator of the PX4 autopilot, by the PX4
//! Development Team. The model is written from its description, not
//! transcribed from its code.
//!
//! @see Yaakov Bar-Shalom, Update with Out-of-Sequence Measurements in
//! Tracking: Exact Solution, IEEE Transactions on Aerospace and Electronic
//! Systems, 38(3), 2002, the exact update of a measurement lagging by less
//! than one update interval, against the reprocessing baseline,
//! https://doi.org/10.1109/TAES.2002.1039398
//! @see
//! https://github.com/PX4/PX4-Autopilot/tree/main/src/modules/vision_target_estimator
//!
//! @details A vehicle tracks a target ahead of it along one axis. The filter
//! estimates the target's position [m] and velocity [m.s^-1] relative to the
//! vehicle, predicted every 0.02 s from the vehicle's measured acceleration
//! [m.s^-2], the input: the target moves at a nearly constant velocity while
//! the vehicle sways back and forth, so that the relative velocity keeps
//! changing. A camera measures the relative position every 0.1 s with a noise
//! of 0.1 m, but its image processing delivers each measurement 0.1 s, five
//! predictions, after the instant it describes:
//!
//! @image html kf_2x1x1_delayed_target_tracking_unit_layout.svg
//!
//! Fused when it arrives, a measurement of where the target was is taken for
//! where it is: the estimate lags the relative motion by the delay, more than
//! the measurement noise. The sample instead keeps the track, the estimate
//! uncertainty and the input of the last predictions: a delayed measurement
//! rewinds the filter to the instant it describes, through the state and
//! estimate uncertainty setters, updates it there, then replays the later
//! predictions with their recorded inputs:
//!
//! @image html kf_2x1x1_delayed_target_tracking_unit_timeline.svg
//!
//! For a linear filter, the reprocessing is exact: once a measurement has
//! arrived, the estimate is the one a filter given the same measurements
//! without delay would have. The quantities are typed: the history holds a
//! track, its covariance, and an acceleration, so that a rewind can only
//! restore a state and an estimate uncertainty of the filter's own types.
//! Reprocessing is the exact, general baseline of the out-of-sequence
//! measurement literature, at the cost of a history and a replay. The cited
//! update of Bar-Shalom reaches the same estimate without either, for a
//! measurement lagging by less than one update interval, by retrodicting the
//! state to the measurement's instant; the sample keeps the reprocessing.
//!
//! The vehicle's velocity sways between 0 and 8 m.s^-1. Fused as it arrives, a
//! measurement leaves the estimate behind the target by the distance the
//! target moved during the delay: up to 0.8 m when the relative velocity
//! peaks. In the plotted run, with the target's random walk and the noise, the
//! late error reaches 0.9 m, 0.49 m root mean square. The late filter is also
//! inconsistent: its estimate uncertainty ignores the delay, and its error
//! exceeds three of its 0.05 m standard deviations two thirds of the time. Its
//! velocity estimate lags the same way, by about the vehicle's acceleration
//! times the delay. Rewound, the estimate stays within the measurement noise,
//! 0.06 m root mean square, and its error exceeds three of its standard
//! deviations at 0.4% of the measurements, as a consistent filter's would:
//!
//! @image html kf_2x1x1_delayed_target_tracking_unit.svg
//!
//! @todo Support delayed measurements, with a history of the states, estimate
//! uncertainties and prediction arguments kept by the filter.
//!
//! @example kf_2x1x1_delayed_target_tracking_unit.cpp
[[maybe_unused]] const auto sample{[] -> int {
  constexpr int measurement_period{5}; // predictions
  constexpr int measurement_delay{5};  // predictions
  const auto measurement_deviation{0.1 * m};
  const auto acceleration_deviation{0.05 * m / s2};
  // The target's velocity random walk, its variance growth rate.
  const auto target_velocity_diffusion{0.01 * m2 / s3};
  // The vehicle's sway: its acceleration amplitude and angular frequency.
  const auto sway_acceleration{2. * m / s2};
  const auto sway_frequency{0.5 * rad / s};

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

  // The constant velocity model driven by the vehicle's acceleration: the
  // relative acceleration is its opposite. The process noise is the
  // accelerometer's, integrated over a step, and the target's velocity
  // random walk.
  const auto acceleration_variance{pow<2>(acceleration_deviation)};
  const auto make_filter{[&] -> auto {
    return kalman{
        // The state X: the relative position [m] and velocity [m.s^-1].
        state{track{position{10. * m}, velocity{0. * m / s}}},
        // The output Z: the measured relative position [m].
        output<sighting>,
        // The input U: the vehicle's measured acceleration [m.s^-2].
        input<acceleration>,
        // The estimate uncertainty P [m^2, m^2.s^-2]:
        estimate_uncertainty{[]() -> track_covariance {
          track_covariance value;
          value.at<0, 0>(4. * m2);
          value.at<1, 1>(4. * m2 / s2);
          return value;
        }()},
        // The process uncertainty Q:
        process_uncertainty{[&]() -> track_covariance {
          track_covariance value;
          value.at<0, 0>(acceleration_variance * pow<4>(time_step) / 4.);
          value.at<0, 1>(acceleration_variance * pow<3>(time_step) / 2.);
          value.at<1, 0>(acceleration_variance * pow<3>(time_step) / 2.);
          value.at<1, 1>((acceleration_variance * pow<2>(time_step)) +
                         (target_velocity_diffusion * time_step));
          return value;
        }()},
        // The output uncertainty R [m^2]:
        output_uncertainty{pow<2>(measurement_deviation)},
        // The output model H:
        output_model{[]() -> sighting_model {
          sighting_model value;
          value.at<0>(1.);
          return value;
        }()},
        // The state transition F:
        state_transition{[]() -> track_transition {
          track_transition value{track_transition::matrix::Identity()};
          value.at<0, 1>(time_step);
          return value;
        }()},
        // The input control G: the relative motion is opposite the vehicle's.
        input_control{[]() -> acceleration_control {
          acceleration_control value;
          value.at<0>(-pow<2>(time_step) / 2.);
          value.at<1>(-time_step);
          return value;
        }()}};
  }};

  // The filter fusing each measurement as it arrives, and the one rewinding to
  // its instant.
  auto late{make_filter()};
  auto rewound{make_filter()};

  // The history of the rewound filter: the track and estimate uncertainty
  // after each of the last predictions, and the input of each.
  struct snapshot {
    track estimate;
    track_covariance covariance;
    acceleration measured_acceleration;
  };
  // The snapshots of a measurement's instant through the current one.
  constexpr std::size_t history_size{measurement_delay + 1};
  std::array<snapshot, history_size> history{};

  // The camera's measurements in flight, the relative positions of the
  // instants they describe, within a delay through the current one.
  constexpr std::size_t in_flight_size{
      (measurement_delay / measurement_period) + 1};
  std::array<position, in_flight_size> in_flight{};

  position true_position{10. * m};
  velocity target_velocity{0. * m / s};
  squared_position late_squared_errors{0. * m2};
  squared_position rewound_squared_errors{0. * m2};
  int fused{0};
  const int steps{3000};
  const int settled{250};
  for (int step{1}; step <= steps; ++step) {
    const auto phase{sway_frequency * (step * time_step)};
    const acceleration vehicle_acceleration{sway_acceleration *
                                            mp_units::si::sin(phase)};
    target_velocity += sqrt(target_velocity_diffusion * time_step) * noise();
    const velocity true_velocity{target_velocity -
                                 (sway_acceleration / sway_frequency * rad *
                                  (1. - mp_units::si::cos(phase)))};
    true_position += true_velocity * time_step;
    const acceleration measured_acceleration{
        vehicle_acceleration + (acceleration_deviation * noise())};

    late.predict(measured_acceleration);
    rewound.predict(measured_acceleration);
    history.at(static_cast<std::size_t>(step) %
               history_size) = {.estimate = rewound.x(),
                                .covariance = rewound.p(),
                                .measured_acceleration = measured_acceleration};

    // The camera describes this instant, and delivers it later.
    if (step % measurement_period == 0) {
      in_flight.at(static_cast<std::size_t>(step / measurement_period) %
                   in_flight_size) =
          true_position + (measurement_deviation * noise());
    }

    // A measurement arrives, describing the step one delay earlier.
    if (const int described{step - measurement_delay};
        described > 0 && described % measurement_period == 0) {
      const position arrived{in_flight.at(
          static_cast<std::size_t>(described / measurement_period) %
          in_flight_size)};

      late.update(arrived);

      // Rewind to the measurement's instant, update there, then replay the
      // predictions since, refreshing their history.
      const snapshot &then{
          history.at(static_cast<std::size_t>(described) % history_size)};
      rewound.x(then.estimate);
      rewound.p(then.covariance);
      rewound.update(arrived);
      for (int replayed{described + 1}; replayed <= step; ++replayed) {
        snapshot &entry{
            history.at(static_cast<std::size_t>(replayed) % history_size)};
        rewound.predict(entry.measured_acceleration);
        entry.estimate = rewound.x();
        entry.covariance = rewound.p();
      }
      ++fused;
    }

    // After the first seconds, the error of each estimate.
    if (step > settled) {
      late_squared_errors += pow<2>(late.x().at<0>() - true_position);
      rewound_squared_errors += pow<2>(rewound.x().at<0>() - true_position);
    }
  }

  // The root mean square errors of the estimates.
  const position late_error{sqrt(late_squared_errors / (steps - settled))};
  const position rewound_error{
      sqrt(rewound_squared_errors / (steps - settled))};

  // The rewound filter tracks the target within the measurement noise; the
  // late one lags it, whatever the seed of the noise.
  assert(fused == 599);
  assert(rewound_error < 0.1 * m &&
         "The rewound estimate stays near the target.");
  assert(late_error > 4. * rewound_error &&
         "Fusing a delayed measurement as current lags the target.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
