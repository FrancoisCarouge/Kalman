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

#include <algorithm>
#include <cassert>
#include <random>

#include <mp-units/math.h>
#include <mp-units/systems/isq/space_and_time.h>
#include <mp-units/systems/si.h>
#include <mp-units/systems/si/math.h>
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
using mp_units::si::unit_symbols::s3;

//! @brief A velocity in the north, east, and down frame, a vector [m.s^-1].
using velocity =
    quantity<ned_velocity[m / s], mp_units::utility::cartesian_vector<double>>;
//! @brief A speed, the quantity common to the velocities of all axes
//! [m.s^-1].
using speed = quantity<mp_units::isq::speed[m / s]>;
//! @brief The heading: the angle of the aircraft's nose about the down axis,
//! from the north [rad].
using heading = quantity<mp_units::isq::angular_measure[rad]>;
//! @brief The true airspeed, the magnitude of the air velocity [m.s^-1].
using airspeed = quantity<true_airspeed[m / s]>;
//! @brief The scale factor of the airspeed sensor [1].
using scale_factor = quantity<airspeed_scale[mp_units::one]>;

template <typename... Types> using vector = column_vector<double, Types...>;
//! @brief The air data, the state X: the wind, by its north and east
//! components [m.s^-1], and the scale factor of the airspeed sensor [1].
using air_data = vector<quantity<north_velocity[m / s]>,
                        quantity<east_velocity[m / s]>, scale_factor>;
//! @brief The reading, the output Z: the indicated airspeed [m.s^-1] and the
//! sideslip [1].
using reading = vector<quantity<indicated_airspeed[m / s]>,
                       quantity<sideslip[mp_units::one]>>;
//! @brief The derivatives of the reading by the air data, the output model H.
//! The element of a row and a column is the product of the reading element of
//! the row by the reciprocal of the air data element of the column.
using reading_jacobian = kf::evaluate<kf::quotient<reading, air_data>>;
//! @brief A covariance of the air data, such as the estimate uncertainty P:
//! the element of a row and a column is the product of the air data elements
//! they index.
using air_data_covariance = kf::ᴀʙᵀ<air_data, air_data>;
//! @brief A covariance of the reading, such as the output uncertainty R: the
//! element of a row and a column is the product of the reading elements they
//! index.
using reading_covariance = kf::ᴀʙᵀ<reading, reading>;

//! @brief The air velocity of the aircraft: its ground velocity less the wind.
const auto air_velocity_of{
    [](const air_data &estimate, const velocity &ground) -> velocity {
      return ground - velocity{estimate.at<0>(), estimate.at<1>(),
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
//! the sine of the heading. Each component is restated as a speed, their
//! common quantity, whose rotated sums are restated as the body velocities.
const auto to_body{
    [](const velocity &air, const heading &nose) -> body_air_velocity {
      const auto [north, east, down]{air};
      const speed along_north{north};
      const speed along_east{east};
      const auto cosine{mp_units::si::cos(nose)};
      const auto sine{mp_units::si::sin(nose)};
      return {.forward = forward_velocity(
                  speed{(cosine * along_north) + (sine * along_east)}),
              .right = right_velocity(
                  speed{(-sine * along_north) + (cosine * along_east)})};
    }};

//! @brief The observation h(X): the airspeed sensor reads the scale times the
//! true airspeed, the magnitude of the air velocity, and the sideslip is the
//! right air velocity over the forward one.
const auto observe{[](const air_data &estimate, const velocity &ground,
                      const heading &nose) -> reading {
  const velocity air{air_velocity_of(estimate, ground)};
  const airspeed true_speed{true_airspeed(air.magnitude())};
  const body_air_velocity body{to_body(air, nose)};
  return reading{estimate.at<2>() * true_speed, body.right / body.forward};
}};

//! @brief The output model H = ∂h/∂X: the derivative of the observation by
//! the state, evaluated at the estimate.
//!
//! @details The derivative of the airspeed by the scale is the true airspeed.
//! The derivatives of the airspeed by the wind are the opposite of the scale
//! times the unit vector of the air velocity. The derivatives of the sideslip
//! by the wind combine the body velocities, as the rotation into the body frame
//! does. The ratios of the speeds are restated as the derivatives.
const auto observation_jacobian{[](const air_data &estimate,
                                   const velocity &ground,
                                   const heading &nose) -> reading_jacobian {
  const velocity air{air_velocity_of(estimate, ground)};
  const airspeed true_speed{true_airspeed(air.magnitude())};
  const body_air_velocity body{to_body(air, nose)};
  const auto [north, east, down]{air};
  const speed magnitude{true_speed};
  const speed along_north{north};
  const speed along_east{east};
  const speed forward{body.forward};
  const speed right{body.right};
  const auto cosine{mp_units::si::cos(nose)};
  const auto sine{mp_units::si::sin(nose)};
  const auto scale{estimate.at<2>()};
  reading_jacobian value;
  value.at<0, 0>((indicated_airspeed /
                  north_velocity)(-speed{scale * along_north} / magnitude));
  value.at<0, 1>((indicated_airspeed /
                  east_velocity)(-speed{scale * along_east} / magnitude));
  value.at<0, 2>(reading_jacobian::element<0, 2>{true_speed});
  value.at<1, 0>((sideslip / north_velocity)(
      ((sine * forward) + (cosine * right)) / (forward * forward)));
  value.at<1, 1>((sideslip / east_velocity)(
      ((-cosine * forward) + (sine * right)) / (forward * forward)));
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
//! @see Yaakov Bar-Shalom, X. Rong Li, Thiagalingam Kirubarajan, Estimation
//! with Applications to Tracking and Navigation, John Wiley & Sons, 2001, the
//! normalized innovation squared consistency test,
//! https://doi.org/10.1002/0471221279
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
//! filter: its innovation must lie within five of its standard deviations, a
//! gate on its normalized innovation squared. Every tenth airspeed is corrupted
//! by a spike of 30 m.s^-1, as from a blocked or wet pitot tube, which the gate
//! rejects, and the sideslip of the same update with it. The scale factor is
//! kept within plausible bounds after each update. On a straight flight, the
//! airspeed alone cannot tell a headwind from a scale error: the turn reveals
//! the wind from every side, and within a third of the first circle, some
//! twenty seconds, the wind and the scale factor converge together.
//!
//! In the plotted run, the wind settles within 0.2 m.s^-1 and the scale
//! factor within a percent after 18 s. After the first circle, their root mean
//! square errors, 0.05 m.s^-1 and 0.001, are those of their estimated standard
//! deviations, 0.05 m.s^-1 and 0.001. Over a thousand seeds of the noise, no
//! spike is fused, and three runs reject one consistent update.
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
  const airspeed true_speed{20. * true_airspeed[m / s]};
  const quantity<north_velocity[m / s]> wind_north{north_velocity(4. * m / s)};
  const quantity<east_velocity[m / s]> wind_east{east_velocity(3. * m / s)};
  const scale_factor scale{0.9 * airspeed_scale[mp_units::one]};
  const auto spike_height{30. * indicated_airspeed[m / s]};
  const auto airspeed_deviation{0.5 * indicated_airspeed[m / s]};
  const auto sideslip_deviation{0.05 * sideslip[mp_units::one]};
  // The random walks of the wind, 0.01 m.s^-1, and of the scale factor, 1e-4,
  // per square root of second: their variance growth rates.
  const auto wind_diffusion{0.01 * 0.01 * m2 / s3};
  const auto scale_diffusion{1e-4 * 1e-4 / s};
  const double gate{5.}; // standard deviations

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
      // The state X: no wind and a unit scale factor.
      state{air_data{north_velocity(0. * m / s), east_velocity(0. * m / s),
                     1. * airspeed_scale[mp_units::one]}},
      // The output Z: the indicated airspeed [m.s^-1] and the sideslip [1].
      output<reading>,
      // The estimate uncertainty P:
      estimate_uncertainty{[]() -> air_data_covariance {
        air_data_covariance value;
        value.at<0, 0>(25. * m2 / s2);
        value.at<1, 1>(25. * m2 / s2);
        value.at<2, 2>(0.1 * 0.1 * mp_units::one);
        return value;
      }()},
      // The process uncertainty Q, of the random walks over a step:
      process_uncertainty{[&]() -> air_data_covariance {
        air_data_covariance value;
        value.at<0, 0>(wind_diffusion * time_step);
        value.at<1, 1>(wind_diffusion * time_step);
        value.at<2, 2>(scale_diffusion * time_step);
        return value;
      }()},
      // The output uncertainty R:
      output_uncertainty{[&]() -> reading_covariance {
        reading_covariance value;
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
      update_types<velocity, heading>};

  // The consistency test of the observations, computed outside of the filter:
  // each innovation within its gate.
  const auto consistent{[&filter, gate](const velocity &ground,
                                        const heading &nose,
                                        const reading &measured) -> bool {
    const reading innovation{measured - observe(filter.x(), ground, nose)};
    const reading_jacobian jacobian{
        observation_jacobian(filter.x(), ground, nose)};
    const reading_covariance innovation_uncertainty{
        (jacobian * filter.p() * transposed(jacobian)) + filter.r()};
    return innovation.at<0>() * innovation.at<0>() <
               gate * gate * innovation_uncertainty.at<0, 0>() &&
           innovation.at<1>() * innovation.at<1>() <
               gate * gate * innovation_uncertainty.at<1, 1>();
  }};

  int spikes_fused{0};
  int consistent_rejected{0};
  heading nose{0. * rad};
  const int steps{3000};
  for (int step{0}; step < steps; ++step) {
    // The heading is that of the air velocity: the flight is coordinated.
    const speed air_speed{true_speed};
    const velocity ground{
        north_velocity(air_speed * mp_units::si::cos(nose)) + wind_north,
        east_velocity(air_speed * mp_units::si::sin(nose)) + wind_east,
        down_velocity(0. * m / s)};
    // Every tenth step, an airspeed corrupted by a spike.
    const bool spike{step % 10 == 5};
    const reading measured{
        indicated_airspeed(scale * true_speed) +
            (airspeed_deviation * noise()) +
            (spike ? spike_height : 0. * indicated_airspeed[m / s]),
        sideslip_deviation * noise()};

    filter.predict();

    if (consistent(ground, nose, measured)) {
      filter.update(ground, nose, measured);

      // Keep the scale factor within its bounds.
      air_data bounded{filter.x()};
      bounded.at<2>(std::clamp(bounded.at<2>(),
                               0.5 * airspeed_scale[mp_units::one],
                               1.5 * airspeed_scale[mp_units::one]));
      filter.x(bounded);

      spikes_fused += static_cast<int>(spike);
    } else {
      consistent_rejected += static_cast<int>(!spike);
    }

    nose += turn_rate * time_step;
  }

  // The wind and the scale factor are found, the spikes rejected, whatever the
  // seed of the noise. At five standard deviations, a consistent update is
  // rejected about once in three hundred runs.
  assert(abs(filter.x().at<0>() - wind_north) < 0.25 * m / s &&
         abs(filter.x().at<1>() - wind_east) < 0.25 * m / s &&
         "The wind is found within a quarter of a meter per second.");
  assert(abs(filter.x().at<2>() - scale) < 0.01 * mp_units::one &&
         "The airspeed scale factor is found within a percent.");
  assert(spikes_fused == 0 && "The spikes are rejected.");
  assert(consistent_rejected <= 2 &&
         "Nearly every consistent update is fused.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
