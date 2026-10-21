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
#include <cstddef>
#include <numbers>
#include <utility>

#include <mp-units/compat_macros.h>
#include <mp-units/math.h>
#include <mp-units/systems/isq/space_and_time.h>
#include <mp-units/systems/si.h>
#include <mp-units/systems/si/math.h>

namespace fcarouge::sample {
namespace {
//! @name The North, East, and Down Frame
//! @brief The vector quantities of the local frame. Each axis is a quantity
//! kind of its own: a north component is never added to, nor substituted for,
//! an east component.
//! @{
QUANTITY_SPEC(ned_rotation, mp_units::isq::angular_displacement,
              mp_units::quantity_tensor_order::vector);
QUANTITY_SPEC(north_angle, ned_rotation, mp_units::is_kind);
QUANTITY_SPEC(east_angle, ned_rotation, mp_units::is_kind);
QUANTITY_SPEC(down_angle, ned_rotation, mp_units::is_kind);
QUANTITY_SPEC(ned_velocity, mp_units::isq::velocity);
QUANTITY_SPEC(north_velocity, mp_units::isq::velocity, mp_units::is_kind);
QUANTITY_SPEC(east_velocity, mp_units::isq::velocity, mp_units::is_kind);
QUANTITY_SPEC(down_velocity, mp_units::isq::velocity, mp_units::is_kind);
QUANTITY_SPEC(ned_acceleration, ned_velocity / mp_units::isq::duration);
//! @}

//! @name The Body Frame
//! @brief The vector quantities of the camera's forward, right, and down
//! axes, about which it rolls, pitches, and yaws.
//! @{
QUANTITY_SPEC(body_rotation, mp_units::isq::angular_displacement,
              mp_units::quantity_tensor_order::vector);
QUANTITY_SPEC(body_angular_velocity, body_rotation / mp_units::isq::duration);
QUANTITY_SPEC(roll_rate, body_angular_velocity, mp_units::is_kind);
QUANTITY_SPEC(pitch_rate, body_angular_velocity, mp_units::is_kind);
QUANTITY_SPEC(yaw_rate, body_angular_velocity, mp_units::is_kind);
QUANTITY_SPEC(body_velocity, mp_units::isq::velocity);
//! @}
} // namespace
} // namespace fcarouge::sample

//! @brief The decomposition of a rotation vector in the north, east, and down
//! frame into its axes, in coordinate order.
template <>
struct mp_units::vector_components<fcarouge::sample::ned_rotation>
    : mp_units::vector_axes<fcarouge::sample::north_angle,
                            fcarouge::sample::east_angle,
                            fcarouge::sample::down_angle> {};

//! @brief The decomposition of a north, east, and down velocity into its axes,
//! in coordinate order.
template <>
struct mp_units::vector_components<fcarouge::sample::ned_velocity>
    : mp_units::vector_axes<fcarouge::sample::north_velocity,
                            fcarouge::sample::east_velocity,
                            fcarouge::sample::down_velocity> {};

//! @brief The decomposition of a body angular velocity into its axes, in
//! coordinate order.
template <>
struct mp_units::vector_components<fcarouge::sample::body_angular_velocity>
    : mp_units::vector_axes<fcarouge::sample::roll_rate,
                            fcarouge::sample::pitch_rate,
                            fcarouge::sample::yaw_rate> {};

namespace fcarouge::sample {
namespace {
using mp_units::quantity;
using mp_units::si::unit_symbols::m;
using mp_units::si::unit_symbols::m2;
using mp_units::si::unit_symbols::rad;
using mp_units::si::unit_symbols::s;
using mp_units::si::unit_symbols::s2;
constexpr auto rad2{pow<2>(rad)};

using heading_t = quantity<down_angle[rad]>;
using ned_rotation_t = quantity<ned_rotation[rad], Eigen::Vector3d>;
using ned_velocity_t = quantity<ned_velocity[m / s], Eigen::Vector3d>;
using ned_acceleration_t = quantity<ned_acceleration[m / s2], Eigen::Vector3d>;
using body_rotation_t = quantity<body_rotation[rad], Eigen::Vector3d>;
using body_angular_velocity_t =
    quantity<body_angular_velocity[rad / s], Eigen::Vector3d>;
using body_velocity_t = quantity<body_velocity[m / s], Eigen::Vector3d>;

//! @brief The attitude: the rotation matrix of the body frame into the north,
//! east, and down frame, dimensionless.
//!
//! @details The attitude rotates vector quantities between their frames only
//! through the functions below, which state the frames of their quantities.
using attitude_t = Eigen::Matrix3d;

template <typename... Types> using vector = column_vector<double, Types...>;
//! @brief The state: the errors of the nominal attitude, as a rotation vector
//! in the north, east, and down frame [rad], of the nominal velocity [m.s^-1],
//! and of the gyroscope bias, in the body frame [rad.s^-1].
using state_t =
    vector<quantity<north_angle[rad]>, quantity<east_angle[rad]>,
           quantity<down_angle[rad]>, quantity<north_velocity[m / s]>,
           quantity<east_velocity[m / s]>, quantity<down_velocity[m / s]>,
           quantity<roll_rate[rad / s]>, quantity<pitch_rate[rad / s]>,
           quantity<yaw_rate[rad / s]>>;
//! @brief The output: the errors of the nominal velocity [m.s^-1] and of the
//! nominal heading [rad], the measured values minus the nominal ones.
using output_t =
    vector<quantity<north_velocity[m / s]>, quantity<east_velocity[m / s]>,
           quantity<down_velocity[m / s]>, heading_t>;
using state_transition_t = kf::evaluate<kf::quotient<state_t, state_t>>;
using output_model_t = kf::evaluate<kf::quotient<output_t, state_t>>;
using estimate_uncertainty_t = kf::ᴀʙᵀ<state_t, state_t>;
using output_uncertainty_t = kf::ᴀʙᵀ<output_t, output_t>;

//! @brief The period of the inertial sensors.
constexpr auto time_step{0.01 * s};

//! @brief The rotation matrix of a rotation vector [rad]: the rotation by its
//! magnitude about its direction.
const auto rotation{[](const Eigen::Vector3d &angle) -> attitude_t {
  return Eigen::AngleAxisd{angle.norm(), angle.normalized()}.toRotationMatrix();
}};

//! @brief The attitude after a rotation of the body, in the body frame.
const auto rotate_body{[](const attitude_t &body_to_ned,
                          const body_rotation_t &angle) -> attitude_t {
  return attitude_t{body_to_ned * rotation(angle.numerical_value_in(rad))};
}};

//! @brief The attitude after a rotation in the north, east, and down frame.
const auto rotate_ned{[](const ned_rotation_t &angle,
                         const attitude_t &body_to_ned) -> attitude_t {
  return attitude_t{rotation(angle.numerical_value_in(rad)) * body_to_ned};
}};

//! @brief A body velocity in the north, east, and down frame.
const auto to_ned{[](const attitude_t &body_to_ned,
                     const body_velocity_t &velocity) -> ned_velocity_t {
  return Eigen::Vector3d{body_to_ned * velocity.numerical_value_in(m / s)} *
         ned_velocity[m / s];
}};

//! @brief A north, east, and down velocity in the body frame.
const auto to_body{[](const attitude_t &body_to_ned,
                      const ned_velocity_t &velocity) -> body_velocity_t {
  return Eigen::Vector3d{body_to_ned.transpose() *
                         velocity.numerical_value_in(m / s)} *
         body_velocity[m / s];
}};

//! @brief The heading of the camera: the angle of its forward axis about the
//! down axis, from the north.
const auto heading_of{[](const attitude_t &body_to_ned) -> heading_t {
  return std::atan2(body_to_ned(1, 0), body_to_ned(0, 0)) * down_angle[rad];
}};

//! @brief A heading wrapped within a half turn.
const auto wrap{[](const heading_t &heading) -> heading_t {
  return std::remainder(heading.numerical_value_in(rad),
                        2. * std::numbers::pi) *
         down_angle[rad];
}};

//! @brief The state transition F of the errors over a step: the propagation
//! of the errors through the integration of the nominal values.
//!
//! @details The bias rotates into an attitude error: ∂δθ/∂δb = -R Δt, for the
//! attitude R. An attitude error tilts the specific force into a velocity
//! error: ∂δv/∂δθ = -[(R Δv)×], for the velocity increment Δv of the specific
//! force. The cross product pairs the components of different axes: the north
//! velocity error derives from the east attitude error by the down component of
//! R Δv, the quantity north velocity per east angle, while the component is a
//! down velocity. Their values are equal, their quantities distinct. The value
//! of the component is therefore restated as the derivative.
const auto state_transition_of{
    [](const attitude_t &body_to_ned,
       const body_velocity_t &delta_velocity) -> state_transition_t {
      state_transition_t value{kf::make_one<state_transition_t>()};
      [&body_to_ned,
       &value]<std::size_t... Index>(std::index_sequence<Index...>) -> void {
        (value.at<Index / 3, 6 + (Index % 3)>(
             state_transition_t::element<Index / 3, 6 + (Index % 3)>{
                 -body_to_ned(static_cast<Eigen::Index>(Index / 3),
                              static_cast<Eigen::Index>(Index % 3)) *
                 time_step}),
         ...);
      }(std::make_index_sequence<9>{});
      const auto [north, east, down]{to_ned(body_to_ned, delta_velocity)};
      value.at<3, 1>(down.numerical_value_in(m / s) *
                     (north_velocity / east_angle)[m / s / rad]);
      value.at<3, 2>(-east.numerical_value_in(m / s) *
                     (north_velocity / down_angle)[m / s / rad]);
      value.at<4, 0>(-down.numerical_value_in(m / s) *
                     (east_velocity / north_angle)[m / s / rad]);
      value.at<4, 2>(north.numerical_value_in(m / s) *
                     (east_velocity / down_angle)[m / s / rad]);
      value.at<5, 0>(east.numerical_value_in(m / s) *
                     (down_velocity / north_angle)[m / s / rad]);
      value.at<5, 1>(-north.numerical_value_in(m / s) *
                     (down_velocity / east_angle)[m / s / rad]);
      return value;
    }};

//! @brief Attitude of a camera gimbal and bias of its gyroscope, an
//! error-state extended filter.
//!
//! @copyright This example is inspired by the gimbal attitude filter of the
//! ArduPilot autopilot (SmallEKF), by the ArduPilot Dev Team. The model is
//! written from its equations, not transcribed from its code.
//!
//! @see https://ardupilot.org/copter/docs/common-mounts.html
//! @see
//! https://github.com/ArduPilot/ardupilot/blob/Copter-3.3/libraries/AP_NavEKF/AP_SmallEKF.cpp
//! @see Joan Solà, Quaternion Kinematics for the Error-State Kalman Filter,
//! 2017, https://arxiv.org/abs/1711.02508
//!
//! @details A camera gimbal on a vehicle at rest pans slowly about the down
//! axis while its tilt oscillates. The inertial sensors of the camera measure
//! its rotation and specific force: the gyroscope, offset by an unknown bias,
//! and the accelerometer, a hundred times a second. Ten times a second, a
//! compass measures the heading:
//!
//! @image html ekf_9x4x0_gimbal_attitude_unit_layout.svg
//!
//! The gimbal integrates the gyroscope and accelerometer increments into a
//! nominal attitude and velocity. The filter does not estimate them directly:
//! its nine states are the errors of the nominal values, the attitude error as
//! a rotation vector in the north, east, and down frame, the velocity error,
//! and the gyroscope bias in the body frame. Each axis is a quantity kind of
//! its own. The state transition F is computed from the prediction arguments,
//! the attitude and the velocity increment. The output model H is constant:
//! the vehicle at rest observes a zero velocity, and the heading error of a
//! nearly level camera is the down attitude error. Each correction moves the
//! estimated errors into the nominal values and resets them to zero.
//!
//! The tilt converges within seconds: a tilt error turns gravity into a
//! velocity error, which the zero velocity observes. The heading and the yaw
//! rate bias converge more slowly, as the compass averages out its noise, to
//! a heading error well within the compass noise.
//!
//! The autopilot's filter differs: its nominal attitude is a quaternion, it
//! fuses the vehicle velocity of the main navigation filter, and it aligns
//! its initial heading once the tilt has converged.
//!
//! @image html ekf_9x4x0_gimbal_attitude_unit.svg
//!
//! @todo Support an error-state reset as part of the update, instead of moving
//! the estimated errors out of the state and zeroing it after each update.
//! @todo Support an output type per observation, so that the velocity and the
//! heading can be fused at their own rates.
//!
//! @example ekf_9x4x0_gimbal_attitude_unit.cpp
[[maybe_unused]] const auto sample{[] -> int {
  const ned_acceleration_t gravity{Eigen::Vector3d{0., 0., 9.80665} *
                                   ned_acceleration[m / s2]};
  const body_angular_velocity_t true_bias{roll_rate(0.01 * rad / s),
                                          pitch_rate(-0.02 * rad / s),
                                          yaw_rate(0.005 * rad / s)};
  const auto gyroscope_noise{0.001 * rad / s};
  const auto accelerometer_noise{0.05 * m / s2};
  const auto bias_noise{1e-5 * rad / s}; // per step

  kalman filter{
      // The state X: the errors, zero after each correction.
      state{state_t{}},
      // The output Z: the errors of the velocity and of the heading.
      output<output_t>,
      // The estimate uncertainty P:
      estimate_uncertainty{[]() -> estimate_uncertainty_t {
        estimate_uncertainty_t value;
        value.at<0, 0>(0.01 * rad2);
        value.at<1, 1>(0.01 * rad2);
        value.at<2, 2>(0.01 * rad2);
        value.at<3, 3>(0.01 * m2 / s2);
        value.at<4, 4>(0.01 * m2 / s2);
        value.at<5, 5>(0.01 * m2 / s2);
        value.at<6, 6>(0.0025 * rad2 / s2);
        value.at<7, 7>(0.0025 * rad2 / s2);
        value.at<8, 8>(0.0025 * rad2 / s2);
        return value;
      }()},
      // The process uncertainty Q, of the sensor noises over a step:
      process_uncertainty{[&]() -> estimate_uncertainty_t {
        const auto attitude_variance{pow<2>(gyroscope_noise * time_step)};
        const auto velocity_variance{pow<2>(accelerometer_noise * time_step)};
        estimate_uncertainty_t value;
        value.at<0, 0>(attitude_variance);
        value.at<1, 1>(attitude_variance);
        value.at<2, 2>(attitude_variance);
        value.at<3, 3>(velocity_variance);
        value.at<4, 4>(velocity_variance);
        value.at<5, 5>(velocity_variance);
        value.at<6, 6>(pow<2>(bias_noise));
        value.at<7, 7>(pow<2>(bias_noise));
        value.at<8, 8>(pow<2>(bias_noise));
        return value;
      }()},
      // The output uncertainty R:
      output_uncertainty{[]() -> output_uncertainty_t {
        output_uncertainty_t value;
        value.at<0, 0>(0.01 * m2 / s2);
        value.at<1, 1>(0.01 * m2 / s2);
        value.at<2, 2>(0.01 * m2 / s2);
        value.at<3, 3>(0.0025 * rad2);
        return value;
      }()},
      // The output model H selects the observed errors:
      output_model{[](const state_t &) -> output_model_t {
        output_model_t value;
        value.at<0, 3>(1.);
        value.at<1, 4>(1.);
        value.at<2, 5>(1.);
        value.at<3, 2>(1.);
        return value;
      }},
      // The state transition F, of the attitude and the velocity increment:
      state_transition{state_transition_of},
      // The observation h(X) of the observed errors:
      observation{[](const state_t &errors) -> output_t {
        return output_t{errors.at<3>(), errors.at<4>(), errors.at<5>(),
                        errors.at<2>()};
      }},
      // The filter prediction uses the attitude and the velocity increment.
      prediction_types<attitude_t, body_velocity_t>};

  // The nominal values the errors correct.
  attitude_t body_to_ned{
      rotate_ned(ned_rotation_t{north_angle(0.02 * rad),
                                east_angle(-0.03 * rad), down_angle(0.1 * rad)},
                 attitude_t::Identity())};
  ned_velocity_t velocity{Eigen::Vector3d{Eigen::Vector3d::Zero()} *
                          ned_velocity[m / s]};
  body_angular_velocity_t bias{Eigen::Vector3d{Eigen::Vector3d::Zero()} *
                               body_angular_velocity[rad / s]};

  // Move the estimated errors into the nominal values and reset them.
  const auto reset{[&filter, &body_to_ned, &velocity, &bias] -> void {
    const state_t errors{filter.x()};
    body_to_ned = rotate_ned(
        ned_rotation_t{errors.at<0>(), errors.at<1>(), errors.at<2>()},
        body_to_ned);
    velocity += ned_velocity_t{errors.at<3>(), errors.at<4>(), errors.at<5>()};
    bias +=
        body_angular_velocity_t{errors.at<6>(), errors.at<7>(), errors.at<8>()};
    filter.x(state_t{});
  }};

  attitude_t true_body_to_ned{attitude_t::Identity()};
  const int steps{6000};
  for (int step{0}; step < steps; ++step) {
    const auto time{step * time_step};

    // The true motion: a slow pan with a small oscillation in tilt.
    const body_angular_velocity_t true_rate{
        roll_rate(0.05 * rad / s * mp_units::si::sin(0.3 * rad / s * time)),
        pitch_rate(0.05 * rad / s * mp_units::si::cos(0.2 * rad / s * time)),
        yaw_rate(0.1 * rad / s)};
    const body_rotation_t delta_angle{(true_rate + true_bias) * time_step};
    const body_velocity_t delta_velocity{
        to_body(true_body_to_ned, -gravity * time_step)};
    true_body_to_ned = rotate_body(true_body_to_ned, true_rate * time_step);

    // Predict the errors over the step, then integrate the nominal values.
    filter.predict(body_to_ned, delta_velocity);
    velocity += to_ned(body_to_ned, delta_velocity) + (gravity * time_step);
    body_to_ned = rotate_body(body_to_ned, delta_angle - (bias * time_step));

    // Ten times a second, observe the zero velocity of the vehicle at rest,
    // and the heading of the compass.
    if (step % 10 == 0) {
      const heading_t measured{
          heading_of(true_body_to_ned) +
          (0.05 * std::sin(12.9898 * step) * down_angle[rad])};
      const auto [north, east, down]{velocity};
      filter.update(-north, -east, -down,
                    wrap(measured - heading_of(body_to_ned)));
      reset();
    }
  }

  assert((bias - true_bias).numerical_value_in(rad / s).norm() < 0.001 &&
         "The gyroscope bias is found within a milliradian per second.");
  assert(abs(wrap(heading_of(body_to_ned) - heading_of(true_body_to_ned))) <
             0.01 * rad &&
         "The heading is found within ten milliradians.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
