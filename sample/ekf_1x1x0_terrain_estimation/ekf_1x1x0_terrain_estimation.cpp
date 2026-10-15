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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace fcarouge::sample {
namespace {
//! @brief The sensor of an observation.
enum class sensor : std::uint8_t { range_finder, optical_flow };

//! @brief The predicted observation of a sensor.
//!
//! @details The state is the terrain's vertical position, positive down, in
//! the frame of the vehicle's vertical position. The range finder looks along
//! the vehicle's tilted vertical axis and measures the height above ground
//! divided by the cosine of the tilt [m]. The optical flow sensor measures the
//! apparent angular rate of the ground, the horizontal velocity divided by the
//! range along its axis [rad.s^-1].
[[nodiscard]] auto predict_observation(double terrain, sensor kind,
                                       double position_down, double cos_tilt,
                                       double velocity) -> double {
  const double height_above_ground{terrain - position_down};
  if (kind == sensor::range_finder) {
    return height_above_ground / cos_tilt;
  }
  return velocity * cos_tilt / height_above_ground;
}

//! @brief The derivative of the predicted observation with respect to the
//! terrain's vertical position.
[[nodiscard]] auto observation_jacobian(double terrain, sensor kind,
                                        double position_down, double cos_tilt,
                                        double velocity) -> double {
  const double height_above_ground{terrain - position_down};
  if (kind == sensor::range_finder) {
    return 1 / cos_tilt;
  }
  return -velocity * cos_tilt / (height_above_ground * height_above_ground);
}

//! @brief Terrain height estimation from a range finder and optical flow.
//!
//! @see https://github.com/PX4/PX4-ECL
//! @see
//! https://github.com/PX4/PX4-ECL/tree/master/EKF/matlab/scripts/Terrain%20Estimator
//! @see https://ardupilot.org/copter/docs/common-optical-flow-sensor-setup.html
//!
//! @details A vehicle flying at a constant altitude over rising ground
//! estimates the vertical position of the terrain below it, the one state of
//! the filter, from a downward range finder and an optical flow sensor. The
//! terrain is constant but for a process noise proportional to the distance
//! travelled, from an assumed maximum terrain gradient. The range finder drops
//! out for twenty seconds, during which the optical flow alone, through the
//! known velocity, keeps the estimate. Autopilots such as ArduPilot and PX4 run
//! such an estimator beside their navigation filter; the model here follows
//! the terrain estimator of the PX4 Estimation and Control Library, written
//! from its equations.
//!
//! The two sensors observe the one state in different units, meters and
//! radians per second: the filter is declared with an untyped output, and
//! the output uncertainty R is set before each update for its sensor.
//!
//! @todo Declare an output type per observation, so that the filter can carry
//! the units of each sensor.
//! @todo Expose the innovation and its uncertainty before the correction, so
//! that an observation can be rejected by its consistency test without
//! recomputing them outside of the filter.
//! @todo Support a constraint applied after each update, here keeping the
//! terrain below the vehicle.
//!
//! @example ekf_1x1x0_terrain_estimation.cpp
[[maybe_unused]] const auto sample{[] -> int {
  const double time_step{0.1};                // s
  const double velocity{6.};                  // m.s^-1, north
  const double position_down{-30.};           // m, the vehicle's altitude
  const double terrain_slope{0.03};           // m.m^-1, the ground rises
  const double terrain_gradient_maximum{0.2}; // m.m^-1, assumed
  const double range_variance{0.05 * 0.05};   // m^2
  const double flow_variance{0.01 * 0.01};    // (rad.s^-1)^2
  const double ground_clearance{0.1};         // m
  const double gate{3.};                      // standard deviations

  // The true terrain vertical position below the vehicle at a north distance.
  const auto true_terrain{[terrain_slope](double north) -> double {
    return -(2. + (terrain_slope * north));
  }};

  kalman filter{
      // The state X: the terrain vertical position, initially at the datum.
      state{0.},
      // The output Z: a range [m] or an optical flow rate [rad.s^-1].
      output<double>,
      // The estimate uncertainty P [m^2]:
      estimate_uncertainty{100.},
      // The process uncertainty Q, set before each prediction [m^2]:
      process_uncertainty{0.},
      // The output uncertainty R, set before each update:
      output_uncertainty{0.},
      // The output model H = ∂h/∂X of the sensor:
      output_model{[](const double &terrain, const sensor &kind,
                      const double &vehicle_down, const double &cos_tilt,
                      const double &speed) -> double {
        return observation_jacobian(terrain, kind, vehicle_down, cos_tilt,
                                    speed);
      }},
      // The terrain is constant: the state transition is the identity.
      transition{[](const double &terrain) -> double { return terrain; }},
      // The observation h(X) of the sensor:
      observation{[](const double &terrain, const sensor &kind,
                     const double &vehicle_down, const double &cos_tilt,
                     const double &speed) -> double {
        return predict_observation(terrain, kind, vehicle_down, cos_tilt,
                                   speed);
      }},
      update_types<sensor, double, double, double>, prediction_types<>};

  // The consistency test of an observation, computed outside of the filter.
  const auto consistent{[&filter, gate, position_down,
                         velocity](double measured, sensor kind,
                                   double cos_tilt, double variance) -> bool {
    const double innovation{measured - predict_observation(filter.x(), kind,
                                                           position_down,
                                                           cos_tilt, velocity)};
    const double jacobian{observation_jacobian(filter.x(), kind, position_down,
                                               cos_tilt, velocity)};
    const double innovation_variance{(jacobian * filter.p() * jacobian) +
                                     variance};
    return innovation * innovation < gate * gate * innovation_variance;
  }};

  // Fuse one observation if consistent, then keep the terrain below the
  // vehicle.
  const auto fuse{[&filter, &consistent, ground_clearance, position_down,
                   velocity](double measured, sensor kind, double cos_tilt,
                             double variance) -> void {
    if (!consistent(measured, kind, cos_tilt, variance)) {
      return;
    }
    filter.r(variance);
    filter.update(kind, position_down, cos_tilt, velocity, measured);
    filter.x(std::max(filter.x(), position_down + ground_clearance));
  }};

  const int steps{600};
  for (int step{0}; step < steps; ++step) {
    const double time{step * time_step};
    const double north{velocity * time};
    const double cos_tilt{std::cos(0.05 * std::sin(0.5 * time))};
    const double height_above_ground{true_terrain(north) - position_down};

    // The terrain changes by at most the gradient over the distance flown.
    const double distance{velocity * time_step};
    filter.q(distance * terrain_gradient_maximum * distance *
             terrain_gradient_maximum);
    filter.predict();

    // Deterministic measurement noise, within one standard deviation.
    const double range_noise{0.05 * std::sin(12.9898 * step)};
    const double flow_noise{0.01 * std::sin(78.233 * step)};

    // The range finder drops out between 20 s and 40 s.
    const bool range_available{time < 20. || time >= 40.};
    if (range_available) {
      fuse((height_above_ground / cos_tilt) + range_noise, sensor::range_finder,
           cos_tilt, range_variance);
    }
    fuse((velocity * cos_tilt / height_above_ground) + flow_noise,
         sensor::optical_flow, cos_tilt, flow_variance);

    // The flow keeps the estimate within half a meter through the dropout.
    if (time >= 30. && time < 40.) {
      assert(std::abs(filter.x() - true_terrain(north)) < 0.5 &&
             "The optical flow alone keeps the terrain estimate.");
    }
  }

  const double final_north{velocity * (steps - 1) * time_step};
  assert(std::abs(filter.x() - true_terrain(final_north)) < 0.2 &&
         "The range finder brings the terrain estimate back.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
