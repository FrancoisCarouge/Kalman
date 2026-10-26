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

#include <concepts>
#include <type_traits>

namespace fcarouge::test {
namespace {
//! @test Verifies the policies declared with a filter are part of its type, in
//! their canonical order whatever their order of declaration, their absence and
//! the explicit Joseph form the default filter, and only the floors of a
//! runtime standard deviation taking space.
using simple_type = decltype(kalman{state{0.}, simple_form});
using symmetric_type = decltype(kalman{symmetric_form, state{0.}});
using information_type =
    decltype(kalman{state{0.}, output<double>, information_form});
using conditioned_type = decltype(kalman{state{0.}, output<double>, floors{0.5},
                                         symmetrize, symmetric_form});
using constant_type = decltype(kalman{state{0.}, floors{cw<0.5>}, symmetrize});
using mixed_type = decltype(kalman{state{0.}, floors{0.5}});
using policies_only_type = decltype(kalman{simple_form});

static_assert(std::same_as<decltype(kalman{state{0.}}),
                           kalman<kf::x_z_p_r<double, kf::policies<>>>>);
static_assert(std::same_as<decltype(kalman{state{0.}, joseph_form}),
                           decltype(kalman{state{0.}})>);
static_assert(std::same_as<decltype(kalman{joseph_form, state{0.}, symmetrize}),
                           decltype(kalman{state{0.}, symmetrize})>);
static_assert(
    std::same_as<simple_type,
                 kalman<kf::x_z_p_r<double, kf::policies<kf::simple_form_t>>>>);
static_assert(std::same_as<
              symmetric_type,
              kalman<kf::x_z_p_r<double, kf::policies<kf::symmetric_form_t>>>>);
static_assert(
    std::same_as<
        information_type,
        kalman<kf::x_z_p_r<double, kf::policies<kf::information_form_t>>>>);
static_assert(std::same_as<
              conditioned_type,
              kalman<kf::x_z_p_r<
                  double, kf::policies<kf::symmetric_form_t, kf::symmetrize_t,
                                       kf::floors<double>>>>>);
static_assert(
    std::same_as<conditioned_type,
                 decltype(kalman{symmetric_form, symmetrize, state{0.},
                                 output<double>, floors{0.5}})>);
static_assert(
    std::same_as<
        constant_type,
        kalman<kf::x_z_p_r<
            double,
            kf::policies<kf::symmetrize_t,
                         kf::floors<std::integral_constant<double, 0.5>>>>>>);
static_assert(
    std::same_as<policies_only_type,
                 kalman<kf::x_z_p_r<double, kf::policies<kf::simple_form_t>>>>);
} // namespace
} // namespace fcarouge::test
