/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.4.0
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

#include <mp-units/format.h>
#include <mp-units/framework/quantity.h>
#include <mp-units/math.h>
#include <mp-units/systems/si.h>

namespace fcarouge {
//! @brief The physical unit quantity.
template <typename Representation, auto Reference>
using quantity = mp_units::quantity<Reference, Representation>;

//! @brief The singleton one matrix specialization.
template <typename Representation, auto Reference>
inline constexpr quantity<Representation, Reference>
    one<quantity<Representation, Reference>>{1., Reference};

using mp_units::si::unit_symbols::m;
using mp_units::si::unit_symbols::m2;
using mp_units::si::unit_symbols::s;
using mp_units::si::unit_symbols::s2;
using mp_units::si::unit_symbols::s3;

//! @todo: Consider upstreaming named symbols up to pow<8> because that would be
//! common for constant jerk uncertainties values?
inline constexpr auto s4{pow<4>(s)};

//! @todo Height should be a quantity_point, not a (relative?) quantity?
//! How to deduce filter types? The multiply operator does not make sense for a
//! quantity_point. Deducing the types is not quite correct?
using height = mp_units::quantity<mp_units::isq::height[m]>;
} // namespace fcarouge

#endif // FCAROUGE_UNIT_HPP
