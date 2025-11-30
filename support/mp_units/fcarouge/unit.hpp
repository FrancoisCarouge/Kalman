/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.3
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

#ifndef FCAROUGE_UNIT_HPP
#define FCAROUGE_UNIT_HPP

//! @file
//! @brief Quantities and units facade for mp-units third party implementation.
//!
//! @details Supporting quantities, values, and functions.

#include <mp-units/framework/quantity.h>
#include <mp-units/framework/quantity_point.h>
#include <mp-units/math.h>
#include <mp-units/systems/isq/thermodynamics.h>
#include <mp-units/systems/si.h>

namespace fcarouge {
using mp_units::delta;
using mp_units::point;
using mp_units::si::unit_symbols::deg_C;
using mp_units::si::unit_symbols::m;
using mp_units::si::unit_symbols::m2;
using mp_units::si::unit_symbols::s;
using mp_units::si::unit_symbols::s2;
using mp_units::si::unit_symbols::s3;

inline constexpr auto s4{pow<4>(s)};
inline constexpr auto deg_C2{pow<2>(deg_C)};

using height = mp_units::quantity<mp_units::isq::height[m]>;
using position = mp_units::quantity<mp_units::isq::length[m]>;
using velocity = mp_units::quantity<mp_units::isq::velocity[m / s]>;
using acceleration = mp_units::quantity<mp_units::isq::acceleration[m / s2]>;
using temperature =
    mp_units::quantity_point<mp_units::isq::Celsius_temperature[deg_C]>;

namespace kalman_internal {
template <auto Reference1, auto Reference2>
struct multiplies<mp_units::quantity_point<Reference1>,
                  mp_units::quantity_point<Reference2>> {
  [[nodiscard]] static constexpr auto
  operator()(const mp_units::quantity_point<Reference1> &lhs,
             const mp_units::quantity_point<Reference2> &rhs)
      -> mp_units::quantity<Reference1 * Reference2>;
};

template <typename Representation, auto Reference>
inline constexpr mp_units::quantity<Reference, Representation>
    one<mp_units::quantity<Reference, Representation>>{1., Reference};

template <auto Reference>
inline mp_units::quantity_point<Reference>
    one<mp_units::quantity_point<Reference>>{point<Reference>(1.)};

template <auto Reference>
inline mp_units::quantity_point<Reference>
    zero<mp_units::quantity_point<Reference>>{point<Reference>(0.)};
} // namespace kalman_internal
} // namespace fcarouge

#endif // FCAROUGE_UNIT_HPP
