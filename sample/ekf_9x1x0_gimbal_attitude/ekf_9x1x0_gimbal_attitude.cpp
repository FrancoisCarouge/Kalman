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
#include <numbers>

namespace fcarouge::sample {
namespace {
template <auto Size> using vector = column_vector<double, Size>;
template <auto Row, auto Column> using matrix = matrix<double, Row, Column>;

//! @brief The skew-symmetric matrix of the cross product with a vector.
[[nodiscard]] auto skew(const vector<3> &value) -> matrix<3, 3> {
  return matrix<3, 3>{{0., -value(2), value(1)},
                      {value(2), 0., -value(0)},
                      {-value(1), value(0), 0.}};
}

//! @brief The rotation of a rotation vector, by the Rodrigues formula.
[[nodiscard]] auto rotation(const vector<3> &angle) -> matrix<3, 3> {
  const double magnitude{std::sqrt(
      (angle(0) * angle(0)) + (angle(1) * angle(1)) + (angle(2) * angle(2)))};
  const matrix<3, 3> identity{matrix<3, 3>::Identity()};
  if (magnitude < 1e-12) {
    return identity + skew(angle);
  }
  const matrix<3, 3> axis{skew(angle / magnitude)};
  return identity + (std::sin(magnitude) * axis) +
         ((1 - std::cos(magnitude)) * axis * axis);
}

//! @brief The heading of a body to earth rotation [rad].
[[nodiscard]] auto heading(const matrix<3, 3> &body_to_earth) -> double {
  return std::atan2(body_to_earth(1, 0), body_to_earth(0, 0));
}

//! @brief The state of a scalar observation: the error states 3 to 5 are the
//! velocity errors north, east, and down, the error state 2 approximates the
//! heading error of a level body.
using observed_state = Eigen::Index;

//! @brief Gimbal attitude and gyroscope bias estimation, an error-state filter.
//!
//! @see https://github.com/priseborough/InertialNav
//! @see https://ardupilot.org/copter/docs/common-mounts.html
//!
//! @details A camera gimbal estimates its own attitude from its gyroscope and
//! accelerometer, whose delta angles and delta velocities it integrates into a
//! nominal attitude and velocity. The filter does not estimate them directly:
//! its nine states are the errors of the nominal values, the attitude error as
//! a rotation vector in the earth frame [rad], the velocity error north, east,
//! and down [m.s^-1], and the gyroscope bias [rad.s^-1]. Each prediction
//! propagates the errors through the integration: the bias rotates into an
//! attitude error, and an attitude error tilts the specific force into a
//! velocity error. Each correction moves the estimated errors into the nominal
//! values and resets the errors to zero. The gimbal's vehicle is at rest, so
//! its velocity is observed as zero, one axis at a time, and a compass
//! observes the heading. Autopilots such as ArduPilot run such a filter for
//! their gimbals; the model here is the error-state formulation of the
//! InertialNav project, written from its equations.
//!
//! The observations are a velocity [m.s^-1] and a heading [rad] of the same
//! errors: the filter is declared with an untyped output, and the output
//! uncertainty R is set before each update for its observation.
//!
//! @todo Declare an output type per observation, so that the filter can carry
//! the units of each.
//! @todo Support a state transition F computed from the prediction arguments
//! together with an output model H computed from the update arguments; the
//! transition here is set before each prediction instead.
//! @todo Support an error-state reset as part of the update, instead of moving
//! the estimated errors out of the state and zeroing it after each update.
//!
//! @example ekf_9x1x0_gimbal_attitude.cpp
[[maybe_unused]] const auto sample{[] -> int {
  const double time_step{0.01};                  // s
  const vector<3> gravity{0., 0., 9.80665};      // m.s^-2, down
  const vector<3> true_bias{0.01, -0.02, 0.005}; // rad.s^-1
  const double gyroscope_noise{0.001};           // rad.s^-1
  const double accelerometer_noise{0.05};        // m.s^-2
  const double bias_noise{1e-5};                 // rad.s^-1 per step
  const double velocity_variance{0.1 * 0.1};     // (m.s^-1)^2
  const double heading_variance{0.05 * 0.05};    // rad^2

  matrix<9, 9> process_noise{matrix<9, 9>::Zero()};
  for (Eigen::Index index{0}; index < 3; ++index) {
    process_noise(index, index) =
        gyroscope_noise * gyroscope_noise * time_step * time_step;
    process_noise(3 + index, 3 + index) =
        accelerometer_noise * accelerometer_noise * time_step * time_step;
    process_noise(6 + index, 6 + index) = bias_noise * bias_noise;
  }
  matrix<9, 9> initial_uncertainty{matrix<9, 9>::Zero()};
  for (Eigen::Index index{0}; index < 3; ++index) {
    initial_uncertainty(index, index) = 0.1 * 0.1;
    initial_uncertainty(3 + index, 3 + index) = 0.1 * 0.1;
    initial_uncertainty(6 + index, 6 + index) = 0.05 * 0.05;
  }

  kalman filter{
      // The state X: the errors, zero after each correction.
      state{vector<9>{vector<9>::Zero()}},
      // The output Z: the error of a velocity [m.s^-1] or of the heading
      // [rad], the measured value minus the nominal one.
      output<double>,
      // The estimate uncertainty P:
      estimate_uncertainty{initial_uncertainty},
      // The process uncertainty Q:
      process_uncertainty{process_noise},
      // The output uncertainty R, set before each update:
      output_uncertainty{0.},
      // The output model H selects the observed error state:
      output_model{
          [](const vector<9> &, const observed_state &index) -> matrix<1, 9> {
            matrix<1, 9> model{matrix<1, 9>::Zero()};
            model(0, index) = 1.;
            return model;
          }},
      // The errors are zero at each prediction, after the reset of the last
      // correction: the transition function is the identity, and the state
      // transition F, set before each prediction, propagates the uncertainty.
      transition{[](const vector<9> &errors) -> vector<9> { return errors; }},
      // The observation h(X) is the observed error state:
      observation{[](const vector<9> &errors, const observed_state &index)
                      -> double { return errors(index); }},
      update_types<observed_state>, prediction_types<>};

  // The nominal values the errors correct.
  matrix<3, 3> body_to_earth{rotation(vector<3>{0.02, -0.03, 0.1})};
  vector<3> velocity{vector<3>::Zero()};
  vector<3> bias{vector<3>::Zero()};

  // Move the estimated errors into the nominal values and reset them.
  const auto reset{[&filter, &body_to_earth, &velocity, &bias] -> void {
    const vector<9> errors{filter.x()};
    body_to_earth = rotation(errors.head<3>()) * body_to_earth;
    velocity += errors.segment<3>(3);
    bias += errors.tail<3>();
    filter.x(vector<9>::Zero());
  }};

  matrix<3, 3> true_body_to_earth{matrix<3, 3>::Identity()};
  const int steps{6000};
  for (int step{0}; step < steps; ++step) {
    const double time{step * time_step};

    // The true motion: a slow pan with a small oscillation in tilt.
    const vector<3> true_rate{0.05 * std::sin(0.3 * time),
                              0.05 * std::cos(0.2 * time), 0.1};
    const vector<3> delta_angle{(true_rate + true_bias) * time_step};
    const vector<3> delta_velocity{true_body_to_earth.transpose() * -gravity *
                                   time_step};
    true_body_to_earth = true_body_to_earth * rotation(true_rate * time_step);

    // Predict the errors over the integration step, then integrate the
    // nominal values: the bias rotates into an attitude error, and an attitude
    // error tilts the specific force into a velocity error.
    matrix<9, 9> transition{matrix<9, 9>::Identity()};
    transition.block<3, 3>(0, 6) = -body_to_earth * time_step;
    transition.block<3, 3>(3, 0) = -skew(body_to_earth * delta_velocity);
    filter.f(transition);
    filter.predict();
    velocity += (body_to_earth * delta_velocity) + (gravity * time_step);
    body_to_earth = body_to_earth * rotation(delta_angle - (bias * time_step));

    // The vehicle is at rest: observe each velocity as zero.
    filter.r(velocity_variance);
    for (observed_state axis{3}; axis < 6; ++axis) {
      filter.update(axis, 0. - velocity(axis - 3));
      reset();
    }

    // A compass observes the heading ten times a second.
    if (step % 10 == 0) {
      const double measured{heading(true_body_to_earth) +
                            (0.05 * std::sin(12.9898 * step))};
      double heading_error{measured - heading(body_to_earth)};
      heading_error = std::remainder(heading_error, 2 * std::numbers::pi);
      filter.r(heading_variance);
      filter.update(observed_state{2}, heading_error);
      reset();
    }
  }

  assert((bias - true_bias).norm() < 0.002 &&
         "The gyroscope bias is found within 0.002 rad/s.");
  assert(std::abs(std::remainder(heading(body_to_earth) -
                                     heading(true_body_to_earth),
                                 2 * std::numbers::pi)) < 0.05 &&
         "The heading is found within 0.05 rad.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
