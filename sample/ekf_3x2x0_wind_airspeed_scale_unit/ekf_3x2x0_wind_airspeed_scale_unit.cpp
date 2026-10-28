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

#include <mp-units/systems/isq/space_and_time.h>
#include <mp-units/systems/si.h>
#include <mp-units/utility/cartesian_vector.h>

namespace fcarouge::sample {
namespace {
//! @brief A velocity in the local north, east, and down frame: a vector
//! quantity.
constexpr struct ned_velocity
    : mp_units::quantity_spec<mp_units::isq::velocity> {
} ned_velocity;

//! @name The Axes of the North, East, and Down Frame
//! @brief Each axis is a quantity kind of its own: a north velocity is never
//! added to, nor substituted for, an east velocity.
//! @{
constexpr struct north_velocity
    : mp_units::quantity_spec<mp_units::isq::velocity, mp_units::is_kind> {
} north_velocity;
constexpr struct east_velocity
    : mp_units::quantity_spec<mp_units::isq::velocity, mp_units::is_kind> {
} east_velocity;
constexpr struct down_velocity
    : mp_units::quantity_spec<mp_units::isq::velocity, mp_units::is_kind> {
} down_velocity;
//! @}

//! @name The Axes of the Body Frame
//! @brief The air velocity ahead of the aircraft's nose and to its right.
//! @{
constexpr struct forward_velocity
    : mp_units::quantity_spec<mp_units::isq::velocity, mp_units::is_kind> {
} forward_velocity;
constexpr struct right_velocity
    : mp_units::quantity_spec<mp_units::isq::velocity, mp_units::is_kind> {
} right_velocity;
//! @}

//! @name The Air Data
//! @brief The true airspeed is the magnitude of the air velocity, the
//! indicated airspeed its reading by the airspeed sensor. Their ratio is the
//! scale factor of the sensor. The sideslip is the ratio of the right air
//! velocity to the forward one. Both ratios are dimensionless quantity kinds of
//! their own: a scale factor is never added to, nor substituted for, a
//! sideslip.
//! @{
constexpr struct true_airspeed : mp_units::quantity_spec<mp_units::isq::speed> {
} true_airspeed;
constexpr struct indicated_airspeed
    : mp_units::quantity_spec<mp_units::isq::speed> {
} indicated_airspeed;
constexpr struct airspeed_scale
    : mp_units::quantity_spec<mp_units::dimensionless,
                              indicated_airspeed / true_airspeed,
                              mp_units::is_kind> {
} airspeed_scale;
constexpr struct sideslip
    : mp_units::quantity_spec<mp_units::dimensionless,
                              right_velocity / forward_velocity,
                              mp_units::is_kind> {
} sideslip;
//! @}
} // namespace
} // namespace fcarouge::sample

//! @brief The decomposition of a north, east, and down velocity into its axes,
//! in coordinate order.
template <>
struct mp_units::component_vectors<fcarouge::sample::ned_velocity>
    : mp_units::vector_axes<fcarouge::sample::north_velocity,
                            fcarouge::sample::east_velocity,
                            fcarouge::sample::down_velocity> {};

namespace fcarouge::sample {
namespace {
using mp_units::quantity;
using mp_units::si::unit_symbols::m;
using mp_units::si::unit_symbols::m2;
using mp_units::si::unit_symbols::rad;
using mp_units::si::unit_symbols::s;
using mp_units::si::unit_symbols::s2;

//! @brief A velocity in the north, east, and down frame, a vector [m.s^-1].
using ned_velocity_t =
    quantity<ned_velocity[m / s], mp_units::utility::cartesian_vector<double>>;
//! @brief The heading: the angle of the aircraft's nose about the down axis,
//! from the north [rad].
using heading_t = quantity<mp_units::isq::angular_measure[rad]>;
using true_airspeed_t = quantity<true_airspeed[m / s]>;
using scale_t = quantity<airspeed_scale[mp_units::one]>;

template <typename... Types> using vector = column_vector<double, Types...>;
//! @brief The state: the wind, by its north and east components [m.s^-1], and
//! the scale factor of the airspeed sensor [1].
using state_t = vector<quantity<north_velocity[m / s]>,
                       quantity<east_velocity[m / s]>, scale_t>;
//! @brief The output: the indicated airspeed [m.s^-1] and the sideslip [1].
using output_t = vector<quantity<indicated_airspeed[m / s]>,
                        quantity<sideslip[mp_units::one]>>;
//! @brief The output model H: the derivatives of the output by the state. The
//! element of a row and a column is the product of the output element of the
//! row by the reciprocal of the state element of the column.
using output_model_t = kf::evaluate<kf::quotient<output_t, state_t>>;
//! @brief A covariance of the state: the element of a row and a column is the
//! product of the state elements they index.
using estimate_uncertainty_t = kf::ᴀʙᵀ<state_t, state_t>;
//! @brief A covariance of the output: the element of a row and a column is the
//! product of the output elements they index.
using output_uncertainty_t = kf::ᴀʙᵀ<output_t, output_t>;

//! @brief The air velocity of the aircraft: its ground velocity less the wind.
const auto air_velocity_of{[](const state_t &estimate,
                              const ned_velocity_t &ground) -> ned_velocity_t {
  return ground - ned_velocity_t{estimate.at<0>(), estimate.at<1>(),
                                 down_velocity(0. * m / s)};
}};

//! @brief The air velocity in the body frame, ahead of the nose and to its
//! right.
struct body_air_velocity {
  quantity<forward_velocity[m / s]> forward;
  quantity<right_velocity[m / s]> right;
};

//! @brief The horizontal air velocity rotated by the heading into the body
//! frame.
//!
//! @details The rotation pairs the components of different axes: the forward
//! velocity sums the north and the east velocities, weighted by the cosine and
//! the sine of the heading. Their values add, their quantities do not: the
//! rotation is computed on the values of the components.
const auto to_body{[](const ned_velocity_t &air,
                      const heading_t &heading) -> body_air_velocity {
  const auto [north, east, down]{air};
  const double cosine{std::cos(heading.numerical_value_in(rad))};
  const double sine{std::sin(heading.numerical_value_in(rad))};
  const double air_north{north.numerical_value_in(m / s)};
  const double air_east{east.numerical_value_in(m / s)};
  return {.forward = ((cosine * air_north) + (sine * air_east)) *
                     forward_velocity[m / s],
          .right = ((-sine * air_north) + (cosine * air_east)) *
                   right_velocity[m / s]};
}};

//! @brief The observation h(X): the airspeed sensor reads the scale times the
//! true airspeed, the magnitude of the air velocity, and the sideslip is the
//! right air velocity over the forward one.
const auto observe{[](const state_t &estimate, const ned_velocity_t &ground,
                      const heading_t &heading) -> output_t {
  const ned_velocity_t air{air_velocity_of(estimate, ground)};
  const true_airspeed_t speed{true_airspeed(air.magnitude())};
  const body_air_velocity body{to_body(air, heading)};
  return output_t{estimate.at<2>() * speed, body.right / body.forward};
}};

//! @brief The output model H = ∂h/∂X: the derivative of the observation by
//! the state, evaluated at the estimate.
//!
//! @details The derivative of the airspeed by the scale is the true airspeed.
//! The other derivatives combine the components of different axes, as the
//! rotation into the body frame does: their values are computed on the values
//! of the components, then restated as the derivatives.
const auto observation_jacobian{[](const state_t &estimate,
                                   const ned_velocity_t &ground,
                                   const heading_t &heading) -> output_model_t {
  const ned_velocity_t air{air_velocity_of(estimate, ground)};
  const true_airspeed_t speed{true_airspeed(air.magnitude())};
  const body_air_velocity body{to_body(air, heading)};
  const auto [north, east, down]{air};
  const double scale{estimate.at<2>().numerical_value_in(mp_units::one)};
  const double cosine{std::cos(heading.numerical_value_in(rad))};
  const double sine{std::sin(heading.numerical_value_in(rad))};
  const double forward{body.forward.numerical_value_in(m / s)};
  const double right{body.right.numerical_value_in(m / s)};
  const double forward_squared{forward * forward};
  const double airspeed{speed.numerical_value_in(m / s)};
  output_model_t value;
  value.at<0, 0>(-scale * north.numerical_value_in(m / s) / airspeed *
                 (indicated_airspeed / north_velocity)[mp_units::one]);
  value.at<0, 1>(-scale * east.numerical_value_in(m / s) / airspeed *
                 (indicated_airspeed / east_velocity)[mp_units::one]);
  value.at<0, 2>(output_model_t::element<0, 2>{speed});
  value.at<1, 0>(((sine * forward) + (cosine * right)) / forward_squared *
                 (sideslip / north_velocity)[s / m]);
  value.at<1, 1>(((-cosine * forward) + (sine * right)) / forward_squared *
                 (sideslip / east_velocity)[s / m]);
  return value;
}};

//! @brief Wind and airspeed scale factor of a fixed-wing aircraft, from its
//! airspeed and sideslip.
//!
//! @copyright This example is inspired by the wind estimator of the PX4
//! autopilot's airspeed selector, by the PX4 Dev Team, which estimates the
//! wind and the scale factor of the airspeed sensor from the airspeed and the
//! sideslip. The model is written from its equations, not transcribed from its
//! code.
//!
//! @see https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf.html
//! @see https://github.com/PX4/PX4-Autopilot/tree/main/src/lib/wind_estimator
//! @see https://ardupilot.org/plane/docs/airspeed-calibration.html
//!
//! @details A fixed-wing aircraft flies circles at 20 m.s^-1 through the air
//! in a wind of 4 m.s^-1 from the south and 3 m.s^-1 from the west, measuring
//! its ground velocity and heading. Its airspeed sensor reads 0.9 times the
//! true airspeed, with a noise of 0.5 m.s^-1, and it flies coordinated, the
//! air ahead of its nose: its sideslip, the right air velocity over the
//! forward one, is zero, an observation with an uncertainty of 0.05. The
//! air velocity is the ground velocity less the wind, so that each heading of
//! the circle sees the wind from another side:
//!
//! @image html ekf_3x2x0_wind_airspeed_scale_unit_layout.svg
//!
//! The filter estimates the wind and the scale factor of the airspeed sensor,
//! constant but for a random walk, from both observations, every 0.1 s. The
//! wind is a velocity in the north, east, and down frame, each axis a quantity
//! kind of its own. The two observations are of different quantities, the
//! indicated airspeed [m.s^-1] and the sideslip [1], fused together in a
//! two-element output. The scale factor is the quantity indicated airspeed per
//! true airspeed, so that the scale times the true airspeed is the indicated
//! airspeed, and the derivative of the indicated airspeed by the scale is the
//! true airspeed. Both observations are nonlinear in the wind: the output
//! model H = ∂h/∂X is evaluated at the estimate, from the ground velocity and
//! the heading of the update.
//!
//! Each observation is gated by a consistency test computed outside of the
//! filter: its innovation must lie within five of its standard deviations.
//! Every tenth airspeed is corrupted by a spike of 30 m.s^-1, as from a
//! blocked or wet pitot tube, which the gate rejects, and the sideslip of the
//! same update with it. The scale factor is kept within plausible bounds after
//! each update. On a straight flight, the airspeed alone cannot tell a
//! headwind from a scale error: the turn reveals the wind from every side,
//! and within a third of the first circle, some twenty seconds, the wind and
//! the scale factor converge together.
//!
//! The autopilot's filter differs: it fuses each observation in its own
//! update, at its own rate, and tunes its noises and gates to its sensors.
//!
//! @image html ekf_3x2x0_wind_airspeed_scale_unit.svg
//!
//! @todo Support an output type per observation, so that the airspeed and the
//! sideslip can be fused, and rejected, each in its own update.
//! @todo Expose the innovation and its uncertainty before the correction, so
//! that an observation can be rejected by its consistency test without
//! recomputing them outside of the filter.
//! @todo Support a constraint applied after each update, here bounding the
//! scale factor.
//!
//! @example ekf_3x2x0_wind_airspeed_scale_unit.cpp
[[maybe_unused]] const auto sample{[] -> int {
  const auto time_step{0.1 * s};
  const auto turn_rate{0.1 * rad / s};
  const true_airspeed_t true_speed{20. * true_airspeed[m / s]};
  const quantity<north_velocity[m / s]> wind_north{north_velocity(4. * m / s)};
  const quantity<east_velocity[m / s]> wind_east{east_velocity(3. * m / s)};
  const scale_t scale{0.9 * airspeed_scale[mp_units::one]};
  const auto airspeed_deviation{0.5 * indicated_airspeed[m / s]};
  const auto sideslip_deviation{0.05 * sideslip[mp_units::one]};
  const double gate{5.}; // standard deviations

  // A deterministic standard normal sequence, for repeatable sensor noise: the
  // sum of twelve uniform deviates of a linear congruential generator, minus
  // six.
  auto noise{[seed = std::uint32_t{3}] mutable -> double {
    double sum{0.};
    for (int index{0}; index < 12; ++index) {
      seed = (seed * 1664525U) + 1013904223U;
      sum += static_cast<double>(seed) / 4294967296.;
    }
    return sum - 6.;
  }};

  kalman filter{
      // The state X: no wind and a unit scale factor.
      state{state_t{north_velocity(0. * m / s), east_velocity(0. * m / s),
                    1. * airspeed_scale[mp_units::one]}},
      // The output Z: the indicated airspeed [m.s^-1] and the sideslip [1].
      output<output_t>,
      // The estimate uncertainty P:
      estimate_uncertainty{[]() -> estimate_uncertainty_t {
        estimate_uncertainty_t value;
        value.at<0, 0>(25. * m2 / s2);
        value.at<1, 1>(25. * m2 / s2);
        value.at<2, 2>(0.1 * 0.1 * mp_units::one);
        return value;
      }()},
      // The process uncertainty Q, of the random walks of the wind, 0.01
      // m.s^-1 per square root of second, and of the scale factor, 1e-4 per
      // square root of second, over a step:
      process_uncertainty{[&time_step]() -> estimate_uncertainty_t {
        const double step{time_step.numerical_value_in(s)};
        estimate_uncertainty_t value;
        value.at<0, 0>(0.01 * 0.01 * step * m2 / s2);
        value.at<1, 1>(0.01 * 0.01 * step * m2 / s2);
        value.at<2, 2>(1e-4 * 1e-4 * step * mp_units::one);
        return value;
      }()},
      // The output uncertainty R:
      output_uncertainty{[&]() -> output_uncertainty_t {
        output_uncertainty_t value;
        value.at<0, 0>(airspeed_deviation * airspeed_deviation);
        value.at<1, 1>(sideslip_deviation * sideslip_deviation);
        return value;
      }()},
      // The output model H = ∂h/∂X, of the ground velocity and the heading:
      output_model{observation_jacobian},
      // The wind and the scale factor are constant: their prediction is their
      // last estimate, so no transition function is declared.
      // The observation h(X) of the airspeed and the sideslip:
      observation{observe},
      // The filter update uses the ground velocity and the heading.
      update_types<ned_velocity_t, heading_t>};

  // The consistency test of the observations, computed outside of the filter:
  // each innovation within its gate.
  const auto consistent{[&filter, gate](const ned_velocity_t &ground,
                                        const heading_t &heading,
                                        const output_t &measured) -> bool {
    const output_t innovation{measured - observe(filter.x(), ground, heading)};
    const output_model_t jacobian{
        observation_jacobian(filter.x(), ground, heading)};
    const output_uncertainty_t innovation_uncertainty{
        (jacobian * filter.p() * transposed(jacobian)) + filter.r()};
    return innovation.at<0>() * innovation.at<0>() <
               gate * gate * innovation_uncertainty.at<0, 0>() &&
           innovation.at<1>() * innovation.at<1>() <
               gate * gate * innovation_uncertainty.at<1, 1>();
  }};

  int rejected{0};
  const int steps{3000};
  for (int step{0}; step < steps; ++step) {
    const auto time{step * time_step};
    // The heading is that of the air velocity: the flight is coordinated.
    const heading_t heading{
        std::remainder((turn_rate * time).numerical_value_in(rad),
                       2. * std::numbers::pi) *
        rad};
    const double cosine{std::cos(heading.numerical_value_in(rad))};
    const double sine{std::sin(heading.numerical_value_in(rad))};
    const double speed{true_speed.numerical_value_in(m / s)};
    const ned_velocity_t ground{north_velocity(cosine * speed * m / s) +
                                    wind_north,
                                east_velocity(sine * speed * m / s) + wind_east,
                                down_velocity(0. * m / s)};

    filter.predict();

    // Every tenth step, an airspeed corrupted by a spike, which the gate
    // rejects.
    const auto spike{(step % 10 == 5 ? 30. : 0.) * indicated_airspeed[m / s]};
    const output_t measured{indicated_airspeed(scale * true_speed) +
                                (airspeed_deviation * noise()) + spike,
                            sideslip_deviation * noise()};

    if (!consistent(ground, heading, measured)) {
      ++rejected;
      continue;
    }
    filter.update(ground, heading, measured);

    // Keep the scale factor within its bounds.
    state_t bounded{filter.x()};
    bounded.at<2>(std::clamp(bounded.at<2>(),
                             0.5 * airspeed_scale[mp_units::one],
                             1.5 * airspeed_scale[mp_units::one]));
    filter.x(bounded);
  }

  // The wind and the scale factor are found, the spikes rejected.
  assert(abs(filter.x().at<0>() - wind_north) < 0.5 * m / s &&
         abs(filter.x().at<1>() - wind_east) < 0.5 * m / s &&
         "The wind is found within half a meter per second.");
  assert(abs(filter.x().at<2>() - scale) < 0.01 * mp_units::one &&
         "The airspeed scale factor is found within a percent.");
  assert(rejected == steps / 10 && "Only the spikes are rejected.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
