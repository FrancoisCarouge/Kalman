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
#include <tuple>

#include <mp-units/systems/isq/space_and_time.h>
#include <mp-units/utility/cartesian_vector.h>

namespace fcarouge::sample {
namespace {
//! @brief A displacement in the local north, east, and down frame: a vector
//! quantity.
constexpr struct ned_displacement
    : mp_units::quantity_spec<mp_units::isq::displacement> {
} ned_displacement;

//! @name The Axes of the North, East, and Down Frame
//! @brief Each axis is a quantity kind of its own: a north displacement is
//! never added to, nor substituted for, an east displacement.
//! @{
constexpr struct north
    : mp_units::quantity_spec<mp_units::isq::displacement, mp_units::is_kind> {
} north;
constexpr struct east
    : mp_units::quantity_spec<mp_units::isq::displacement, mp_units::is_kind> {
} east;
constexpr struct down
    : mp_units::quantity_spec<mp_units::isq::displacement, mp_units::is_kind> {
} down;
//! @}
} // namespace
} // namespace fcarouge::sample

//! @brief The decomposition of a north, east, and down displacement into its
//! axes, in coordinate order.
template <>
struct mp_units::component_vectors<fcarouge::sample::ned_displacement>
    : mp_units::vector_axes<fcarouge::sample::north, fcarouge::sample::east,
                            fcarouge::sample::down> {};

namespace fcarouge::sample {
namespace {
using mp_units::si::unit_symbols::m;
using mp_units::si::unit_symbols::m2;
using length = mp_units::quantity<mp_units::isq::length[m]>;

//! @brief A position: the displacement from the origin beacon, a vector [m].
using position_t =
    mp_units::quantity<ned_displacement[m],
                       mp_units::utility::cartesian_vector<double>>;

template <typename... Types> using vector = column_vector<double, Types...>;
//! @brief The state: the receiver's position, by its north, east, and down
//! components [m].
using state_t =
    vector<mp_units::quantity<north[m]>, mp_units::quantity<east[m]>,
           mp_units::quantity<down[m]>>;

using output_t = vector<length>;
//! @brief The output model H: the derivatives of the range by the state
//! components. The element of the output row and a state column is the product
//! of the output element by the reciprocal of the state element [m/m].
using output_model_t =
    matrix<double, std::tuple<length>,
           std::tuple<mp_units::quantity<mp_units::one / north[m]>,
                      mp_units::quantity<mp_units::one / east[m]>,
                      mp_units::quantity<mp_units::one / down[m]>>>;
//! @brief A covariance of the state: the element of a row and a column is the
//! product of the state elements they index [m^2].
using covariance_t = matrix<double, state_t, state_t>;

//! @brief The range from a point to a beacon: the magnitude of their
//! displacement [m].
const auto beacon_range{
    [](const position_t &point, const position_t &beacon) -> length {
      return (point - beacon).magnitude();
    }};

//! @brief The derivative of the range with respect to the point: the
//! components of the unit vector from the beacon to the point.
//!
//! @details A component of the unit vector, an axis displacement by the range,
//! is the quantity north/length for the north axis, while the derivative of the
//! range by that axis is the reciprocal quantity, length/north: their values
//! are equal, their quantities distinct. The value of the component is
//! therefore restated as the derivative.
const auto beacon_range_jacobian{
    [](const position_t &point, const position_t &beacon) -> output_model_t {
      const position_t displacement{point - beacon};
      const length range{displacement.magnitude()};
      const auto [north_component, east_component,
                  down_component]{displacement};
      output_model_t value;
      value.at<0>((north_component / range).numerical_value_in(mp_units::one) *
                  (mp_units::isq::length / north)[mp_units::one]);
      value.at<1>((east_component / range).numerical_value_in(mp_units::one) *
                  (mp_units::isq::length / east)[mp_units::one]);
      value.at<2>((down_component / range).numerical_value_in(mp_units::one) *
                  (mp_units::isq::length / down)[mp_units::one]);
      return value;
    }};

//! @brief Position of a receiver from its ranges to beacons at known
//! positions.
//!
//! @copyright This example is inspired by the static position alignment of the
//! range beacon fusion of the ArduPilot autopilot's third generation extended
//! Kalman filter (EKF3), by the ArduPilot Dev Team, itself based on the range
//! beacon derivation of Paul Riseborough's InertialNav project. The model is
//! written from the derivation's equations, not transcribed from either code.
//!
//! @see https://ardupilot.org/copter/docs/common-pozyx.html
//! @see
//! https://github.com/ArduPilot/ardupilot/blob/master/libraries/AP_NavEKF3/AP_NavEKF3_RngBcnFusion.cpp
//! @see
//! https://github.com/priseborough/InertialNav/blob/master/derivations/range_beacon.m
//! @see Yaakov Bar-Shalom, X. Rong Li, Thiagalingam Kirubarajan, Estimation
//! with Applications to Tracking and Navigation, John Wiley & Sons, 2001, the
//! normalized innovation squared consistency test,
//! https://doi.org/10.1002/0471221279
//!
//! @details A receiver at rest among four radio beacons at known positions,
//! north, east, and down [m], measures its range to one beacon at a time. As in
//! the ArduPilot setup of the Pozyx beacons, the beacons stand at the corners
//! of a rectangle, numbered along an N pattern from the origin beacon, the line
//! from the origin to the second beacon due north:
//!
//! @image html ekf_3x1x0_beacon_position_unit_layout.svg
//!
//! A position is a vector quantity: the displacement from the origin beacon,
//! decomposed into its north, east, and down axes, each a quantity kind of its
//! own, so that no axis is added to, nor substituted for, another. The state is
//! the receiver's position, by its axis components, constant but for a small
//! process noise. The observation composes the components back into a vector,
//! whose magnitude is the range. The range is nonlinear in the position: the
//! output model H = ∂h/∂X is the unit vector from the beacon to the receiver,
//! evaluated at the estimate. The beacons stand at different heights, so the
//! vertical position is observable, yet less so than the horizontal position:
//! the receiver sees the beacons from nearly level. Early in the alignment, the
//! vertical estimate crosses to the mirror solution below the ground, where the
//! ranges are nearly as consistent, before the geometry of the beacons resolves
//! the ambiguity. Autopilots such as ArduPilot align the position of their
//! navigation filter with such a filter before using the beacons.
//!
//! Until the alignment is complete, after a hundred ranges, every range is
//! fused. Afterwards a range is fused only if consistent with the estimate: its
//! innovation must lie within three of its standard deviations, a gate on the
//! normalized innovation squared. A multipath reflection, measuring a longer
//! path, is rejected.
//!
//! The autopilot's filter differs in its tuning: its initial variances are the
//! mean range, its process noise is larger, its range noise is reported by the
//! beacons, and it estimates a vertical offset of the beacons separately.
//!
//! @image html ekf_3x1x0_beacon_position_unit.svg
//!
//! @todo Expose the innovation and its uncertainty before the correction, so
//! that a range can be rejected by its consistency test without recomputing
//! them outside of the filter.
//! @todo Support a covariance correction limited to the variances, as the
//! autopilot's filter does, which caps each diagonal element of K H P at the
//! variance it corrects.
//!
//! @example ekf_3x1x0_beacon_position_unit.cpp
[[maybe_unused]] const auto sample{[] -> int {
  // The beacons, in the order of their identifiers along the N pattern.
  const position_t beacons[]{{north(0. * m), east(0. * m), down(-1. * m)},
                             {north(10. * m), east(0. * m), down(-3. * m)},
                             {north(0. * m), east(10. * m), down(-2. * m)},
                             {north(10. * m), east(10. * m), down(-1. * m)}};
  const position_t receiver{north(3. * m), east(4. * m), down(-0.5 * m)};
  const auto range_variance{0.1 * 0.1 * m2};
  const double gate{3.}; // standard deviations
  const int alignment_ranges{100};

  kalman filter{
      // The state X: the receiver's position, initially the beacons' centre.
      state{state_t{north(5. * m), east(5. * m), down(-1.75 * m)}},
      // The output Z: a range [m].
      output<output_t>,
      // The estimate uncertainty P [m^2]:
      estimate_uncertainty{[]() -> covariance_t {
        covariance_t value;
        value.at<0, 0>(25. * m2);
        value.at<1, 1>(25. * m2);
        value.at<2, 2>(25. * m2);
        return value;
      }()},
      // The process uncertainty Q [m^2]:
      process_uncertainty{[]() -> covariance_t {
        covariance_t value;
        value.at<0, 0>(1e-4 * m2);
        value.at<1, 1>(1e-4 * m2);
        value.at<2, 2>(1e-4 * m2);
        return value;
      }()},
      // The output uncertainty R [m^2]:
      output_uncertainty{range_variance},
      // The output model H = ∂h/∂X, for the beacon of the range:
      output_model{[](const state_t &position,
                      const position_t &beacon) -> output_model_t {
        return beacon_range_jacobian(position, beacon);
      }},
      // The receiver is at rest: its prediction is its last estimate, so no
      // transition function is declared.
      // The observation h(X), the range to the beacon:
      observation{
          [](const state_t &position, const position_t &beacon) -> output_t {
            return output_t{beacon_range(position, beacon)};
          }},
      // The filter update uses the position of the ranged beacon.
      update_types<position_t>};

  bool reflection_rejected{false};
  const int ranges{400};
  for (int index{0}; index < ranges; ++index) {
    const position_t &beacon{beacons[index % 4]};

    // Deterministic measurement noise, within one standard deviation, and one
    // reflection three meters long.
    length measured{beacon_range(receiver, beacon) +
                    (0.1 * std::sin(12.9898 * index) * m)};
    const bool reflection{index == 300};
    if (reflection) {
      measured += 3. * m;
    }

    filter.predict();

    // The consistency test of the range, computed outside of the filter.
    const length innovation{measured - beacon_range(filter.x(), beacon)};
    const output_model_t jacobian{beacon_range_jacobian(filter.x(), beacon)};
    // The one-element H P Hᵀ plus the variance R is the variance quantity S.
    const auto innovation_variance{
        (jacobian * filter.p() * transposed(jacobian)) + range_variance};
    const bool consistent{innovation * innovation <
                          gate * gate * innovation_variance};

    if (index < alignment_ranges || consistent) {
      filter.update(beacon, measured);
    } else if (reflection) {
      reflection_rejected = true;
    }
  }

  assert(reflection_rejected && "The multipath reflection is rejected.");
  assert(beacon_range(filter.x(), receiver) < 0.1 * m &&
         "The receiver position is found within ten centimeters.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
