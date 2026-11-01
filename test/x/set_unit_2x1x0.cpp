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

#include <mp-units/systems/isq/space_and_time.h>

namespace fcarouge::test {
namespace {
using mp_units::si::unit_symbols::km;
using mp_units::si::unit_symbols::m;
using mp_units::si::unit_symbols::s;
using position = mp_units::quantity<mp_units::isq::length[m]>;
using velocity = mp_units::quantity<mp_units::isq::velocity[m / s]>;

template <typename... Types> using vector = column_vector<double, Types...>;
using motion = vector<position, velocity>;

template <typename Filter, typename... Values>
concept has_x =
    requires(Filter filter, Values... values) { filter.x(values...); };

//! @test Verifies the state X member function sets the column vector
//! characteristic from element values initializing each its own element type,
//! the quantity at its position.
using filter = decltype(kalman{state{motion{}}, output<vector<position>>});

static_assert(has_x<filter, motion>);
static_assert(has_x<filter, position, velocity>);
static_assert(
    has_x<filter, mp_units::quantity<mp_units::isq::length[km]>, velocity>);
static_assert(not has_x<filter, velocity, position>);
static_assert(not has_x<filter, double, double>);
} // namespace
} // namespace fcarouge::test
