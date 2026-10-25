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
#include <tuple>

#include <mp-units/math.h>
#include <mp-units/systems/isq/space_and_time.h>
#include <mp-units/systems/si.h>
#include <mp-units/systems/si/math.h>

namespace fcarouge::sample {
namespace {
//! @brief The apparent angular rate of the ground in the optical flow sensor's
//! field of view.
constexpr struct flow_rate
    : mp_units::quantity_spec<mp_units::isq::angular_measure /
                              mp_units::isq::duration> {
} flow_rate;

using mp_units::quantity;
using mp_units::si::unit_symbols::m;
using mp_units::si::unit_symbols::m2;
using mp_units::si::unit_symbols::rad;
using mp_units::si::unit_symbols::s;

//! @brief A vertical position: the altitude above the take-off point [m].
using altitude = quantity<mp_units::isq::altitude[m]>;
//! @brief A height above the terrain [m].
using height = quantity<mp_units::isq::height[m]>;
//! @brief The range finder's length to the terrain, along its axis [m].
using slant_range = quantity<mp_units::isq::length[m]>;
//! @brief A terrain gradient: the rise of the terrain altitude by the length
//! travelled [m/m].
using gradient =
    quantity<(mp_units::isq::altitude / mp_units::isq::length)[mp_units::one]>;
//! @brief The optical flow sensor's apparent rate of the terrain [rad.s^-1].
using optical_flow = quantity<flow_rate[rad / s]>;
//! @brief The vehicle's horizontal speed [m.s^-1].
using ground_speed = quantity<mp_units::isq::speed[m / s]>;
//! @brief The vehicle's tilt, rotating the sensors' axis off the vertical
//! [rad].
using tilt_angle = quantity<mp_units::isq::angular_measure[rad]>;

template <typename... Types> using vector = column_vector<double, Types...>;
//! @brief The state: the terrain altitude [m].
using terrain_altitude = vector<altitude>;
//! @brief The output: the range [m] and the flow rate [rad.s^-1].
using measurement = vector<slant_range, optical_flow>;
//! @brief The variance of the state [m^2]: the one-element estimate
//! uncertainty P and process uncertainty Q.
using variance = quantity<pow<2>(mp_units::isq::altitude)[m2]>;
//! @brief The output model H: the derivatives of the outputs by the state. The
//! element of a row is the product of the output element of the row by the
//! reciprocal of the terrain altitude.
using measurement_jacobian =
    matrix<double, measurement::row_indexes,
           std::tuple<quantity<mp_units::one / mp_units::isq::altitude[m]>>>;
//! @brief The output uncertainty R: the element of a row and a column is the
//! product of the output elements they index.
using measurement_covariance =
    matrix<double, measurement::row_indexes, measurement::row_indexes>;
//! @brief The derivative of the range by the terrain altitude [m/m].
using range_derivative = measurement_jacobian::element<0>;
//! @brief The derivative of the flow rate by the terrain altitude
//! [rad.s^-1/m].
using flow_derivative = measurement_jacobian::element<1>;

//! @brief The sensors observed by an update.
//!
//! @details An unobserved sensor, absent or rejected, has a null row in the
//! output model H: its element of the gain is null, so that its output corrects
//! neither the estimate nor its uncertainty, whatever its value.
struct observed {
  bool range{true};
  bool flow{true};
};

//! @brief The predicted range: the range finder looks along the vehicle's
//! tilted vertical axis, the altitude of the vehicle above the terrain by the
//! cosine of the tilt.
const auto predict_range{[](const altitude &terrain, const altitude &vehicle,
                            const tilt_angle &tilt) -> slant_range {
  return (vehicle - terrain) / mp_units::si::cos(tilt);
}};

//! @brief The derivative of the predicted range by the terrain altitude.
//!
//! @details A rising terrain shortens the range: the derivative is the negative
//! reciprocal of the cosine of the tilt, the quantity length per altitude.
const auto range_jacobian{[](const tilt_angle &tilt) -> range_derivative {
  return -1. / mp_units::si::cos(tilt);
}};

//! @brief The predicted flow rate: the ground seen along the sensor's axis
//! moves at the horizontal speed across the range, an angular rate of the speed
//! by the range.
//!
//! @details An angle is an arc length by its radius: the ratio of the speed by
//! the range is the angular rate subtended. The angle is a quantity kind of its
//! own, never reached implicitly, so the ratio is stated as the flow rate.
const auto predict_flow{[](const altitude &terrain, const altitude &vehicle,
                           const tilt_angle &tilt,
                           const ground_speed &speed) -> optical_flow {
  return flow_rate(speed / predict_range(terrain, vehicle, tilt));
}};

//! @brief The derivative of the predicted flow rate by the terrain altitude.
//!
//! @details A rising terrain shortens the range and quickens the flow: the
//! derivative is the flow rate by the altitude of the vehicle above the
//! terrain, the quantity flow rate per altitude.
const auto flow_jacobian{[](const altitude &terrain, const altitude &vehicle,
                            const tilt_angle &tilt,
                            const ground_speed &speed) -> flow_derivative {
  return predict_flow(terrain, vehicle, tilt, speed) / (vehicle - terrain);
}};

//! @brief Terrain altitude estimation from a range finder and optical flow.
//!
//! @copyright This example is inspired by the terrain estimator of the PX4
//! autopilot's Estimation and Control Library (ECL), by the PX4 Development
//! Team. The model is written from its derivation's equations, not transcribed
//! from its code.
//!
//! @see https://github.com/PX4/PX4-ECL
//! @see
//! https://github.com/PX4/PX4-ECL/tree/master/EKF/matlab/scripts/Terrain%20Estimator
//!
//! @details A vehicle flying north at a constant altitude and speed over rising
//! ground estimates the altitude of the terrain below it, the one state of the
//! filter. A downward range finder measures the distance to the ground along
//! the vehicle's tilted vertical axis, ten times a second, as does an optical
//! flow sensor the apparent angular rate of the ground, the speed by that
//! distance. The range finder drops out for twenty seconds, and a branch in its
//! beam once returns a short range:
//!
//! @image html ekf_1x2x0_terrain_estimation_unit_layout.svg
//!
//! The terrain is constant but for a process noise, the change of the terrain
//! over the distance flown in a step at an assumed maximum gradient. Both
//! observations are nonlinear: their output models H = ∂h/∂X are evaluated at
//! the estimate. Each quantity is typed with its units: the state, a terrain
//! altitude in meters, and the output, a range, a length in meters, and a
//! flow rate, an angular rate in radians per second. The filter fuses both
//! sensors in one update.
//!
//! An output is fused only if its sensor is available and the output consistent
//! with the estimate: its innovation must lie within three of its standard
//! deviations, a gate on the normalized innovation. The update arguments name
//! the observed sensors: the row of an unobserved sensor in the output model H
//! is null, so its element of the gain K is null, and its output, whatever its
//! value, corrects neither the estimate nor its uncertainty. The independent
//! sensors, of a diagonal output uncertainty R, are tested one by one: a
//! rejected range leaves the flow fused. After each update, the terrain is kept
//! below the vehicle.
//! Autopilots such as PX4 and ArduPilot run such an estimator beside their
//! navigation filter, for terrain following and landing.
//!
//! The range finder settles the estimate within a few centimeters. Through the
//! dropout, the flow alone keeps the estimate: its standard deviation grows to
//! a third of a meter, where the process noise alone would grow it to 1.7 m.
//! The estimate lags the rising ground by about 0.2 m at most, as the model
//! holds the terrain constant. The lower the vehicle flies over the rising
//! ground, the faster the flow and the more informative: the standard deviation
//! decreases through the end of the dropout. The short range of the branch,
//! fifty-eight standard deviations off, is rejected.
//!
//! The autopilot's estimator differs: its state is the vertical position
//! positive down, it fuses the range as a height, the tilt applied to the
//! measurement, and the flow on two axes from the vehicle's velocity in the
//! sensor frame, and its process noise adds a terrain noise and the vertical
//! velocity uncertainty.
//!
//! @image html ekf_1x2x0_terrain_estimation_unit.svg
//!
//! @todo Support an update of a subset of the outputs, instead of nulling the
//! rows of the output model of the unobserved sensors.
//! @todo Expose the innovation and its uncertainty before the correction, so
//! that an observation can be rejected by its consistency test without
//! recomputing them outside of the filter.
//! @todo Support a constraint applied after each update, here keeping the
//! terrain below the vehicle.
//!
//! @example ekf_1x2x0_terrain_estimation_unit.cpp
[[maybe_unused]] const auto sample{[] -> int {
  const auto time_step{0.1 * s};
  const ground_speed speed{6. * m / s};
  const altitude vehicle{30. * m};
  // The ground rises north, at a gradient under an assumed maximum.
  const gradient terrain_slope{0.03 * gradient::reference};
  const gradient terrain_gradient_maximum{0.2 * gradient::reference};
  const auto range_deviation{0.05 * m};
  const auto flow_deviation{0.01 * rad / s};
  const height ground_clearance{0.1 * m};
  const double gate{3.}; // standard deviations

  // The terrain change over a step at the maximum gradient.
  const altitude terrain_change{terrain_gradient_maximum * speed * time_step};

  // The true terrain altitude below the vehicle at a time.
  const auto true_terrain{[&](const auto &time) -> altitude {
    return 2. * m + (terrain_slope * speed * time);
  }};

  kalman filter{
      // The state X: the terrain altitude, initially at the take-off point.
      state{terrain_altitude{altitude{0. * m}}},
      // The output Z: the range [m] and the flow rate [rad.s^-1].
      output<measurement>,
      // The estimate uncertainty P [m^2]:
      estimate_uncertainty{variance{100. * m2}},
      // The process uncertainty Q [m^2]:
      process_uncertainty{variance{terrain_change * terrain_change}},
      // The output uncertainty R, of independent sensors:
      output_uncertainty{[&]() -> measurement_covariance {
        measurement_covariance value;
        value.at<0, 0>(range_deviation * range_deviation);
        value.at<1, 1>(flow_deviation * flow_deviation);
        return value;
      }()},
      // The output model H = ∂h/∂X, of the observed sensors:
      output_model{[](const terrain_altitude &terrain,
                      const altitude &vehicle_altitude, const tilt_angle &tilt,
                      const ground_speed &vehicle_speed,
                      const observed &sensors) -> measurement_jacobian {
        measurement_jacobian value;
        if (sensors.range) {
          value.at<0>(range_jacobian(tilt));
        }
        if (sensors.flow) {
          value.at<1>(flow_jacobian(terrain.at<>(), vehicle_altitude, tilt,
                                    vehicle_speed));
        }
        return value;
      }},
      // The terrain is constant: its prediction is its last estimate, so no
      // transition function is declared.
      // The observation h(X), the range and the flow rate:
      observation{[](const terrain_altitude &terrain,
                     const altitude &vehicle_altitude, const tilt_angle &tilt,
                     const ground_speed &vehicle_speed,
                     const observed &) -> measurement {
        return measurement{
            predict_range(terrain.at<>(), vehicle_altitude, tilt),
            predict_flow(terrain.at<>(), vehicle_altitude, tilt,
                         vehicle_speed)};
      }},
      // The filter update uses the vehicle altitude, tilt, and speed, and the
      // observed sensors.
      update_types<altitude, tilt_angle, ground_speed, observed>};

  // The consistency test of an output, its innovation within the gate,
  // computed outside of the filter.
  const auto consistent{[gate, &filter](const auto &innovation,
                                        const auto &derivative,
                                        const auto &output_variance) -> bool {
    // The one-element H P Hᵀ plus R is the innovation variance S.
    const auto innovation_variance{(derivative * filter.p() * derivative) +
                                   output_variance};
    return innovation * innovation < gate * gate * innovation_variance;
  }};

  bool branch_rejected{false};
  const int steps{600};
  for (int step{0}; step < steps; ++step) {
    const auto time{step * time_step};
    const tilt_angle tilt{0.05 * rad * mp_units::si::sin(0.5 * rad / s * time)};
    const altitude terrain{true_terrain(time)};

    filter.predict();

    // Deterministic measurement noise, within one standard deviation.
    const slant_range range_noise{std::sin(12.9898 * step) * range_deviation};
    const optical_flow flow_noise{std::sin(78.233 * step) * flow_deviation};

    // The range finder drops out between 20 s and 40 s, and a branch in its
    // beam returns a range eight meters short at 50 s.
    const bool range_available{time < 20. * s || time >= 40. * s};
    const bool branch{step == 500};
    slant_range range{predict_range(terrain, vehicle, tilt) + range_noise};
    if (branch) {
      range -= 8. * m;
    }
    const optical_flow flow{predict_flow(terrain, vehicle, tilt, speed) +
                            flow_noise};

    // Observe each available and consistent sensor.
    const altitude estimate{filter.x().at<>()};
    const observed sensors{
        .range =
            range_available &&
            consistent(range - predict_range(estimate, vehicle, tilt),
                       range_jacobian(tilt), range_deviation * range_deviation),
        .flow = consistent(flow - predict_flow(estimate, vehicle, tilt, speed),
                           flow_jacobian(estimate, vehicle, tilt, speed),
                           flow_deviation * flow_deviation)};
    branch_rejected = branch_rejected || (branch && !sensors.range);

    // Fuse the observed sensors at once, then keep the terrain below the
    // vehicle.
    filter.update(vehicle, tilt, speed, sensors, range, flow);
    filter.x(terrain_altitude{
        std::min(filter.x().at<>(), vehicle - ground_clearance)});

    // The flow keeps the estimate within half a meter through the dropout.
    if (!range_available) {
      assert(abs(filter.x().at<>() - terrain) < 0.5 * m &&
             "The optical flow alone keeps the terrain estimate.");
      assert(filter.p() < 0.16 * m2 &&
             "The optical flow bounds the terrain standard deviation within "
             "0.4 m, which the process noise alone would grow to 1.7 m.");
    }
  }

  assert(branch_rejected && "The short range of the branch is rejected.");
  assert(abs(filter.x().at<>() - true_terrain((steps - 1) * time_step)) <
             0.2 * m &&
         "The range finder brings the terrain estimate back.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
