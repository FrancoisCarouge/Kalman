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

#include <concepts>
#include <type_traits>

namespace fcarouge::test {
namespace {
template <auto Size> using vector = column_vector<double, Size>;

constexpr auto move{[](const vector<2> &x) -> vector<2> {
  return vector<2>{x[0] + x[1], x[1]};
}};
constexpr auto measure{[](const vector<2> &x) -> double { return x[0]; }};

using move_type = std::remove_const_t<decltype(move)>;
using measure_type = std::remove_const_t<decltype(measure)>;

constexpr auto declare{[](const auto &...policies) -> auto {
  return kalman{state{vector<2>{0., 0.}},
                output<double>,
                estimate_uncertainty{{1., 0.}, {0., 1.}},
                process_uncertainty{{0., 0.}, {0., 0.}},
                output_uncertainty{1.},
                transition{move},
                observation{measure},
                unscented{1., 2., 0.},
                policies...};
}};

using default_type = decltype(declare());
using redraw_type = decltype(declare(redraw_sigma_points));
using reuse_type = decltype(declare(symmetrize, reuse_sigma_points));

//! @test Verifies the unscented filter deduced from a transition and an
//! observation function with the unscented transform parameters, its sigma
//! points policies in their canonical order, the explicit default sigma points
//! drawn anew the default filter, and only the reused sigma points retained.
static_assert(
    std::same_as<default_type,
                 kalman<kf::x_z_p_q_r_tt_oo_ut<vector<2>, double, move_type,
                                               measure_type, double>>>);
static_assert(std::same_as<redraw_type, default_type>);
static_assert(std::same_as<
              reuse_type,
              kalman<kf::x_z_p_q_r_tt_oo_ut<
                  vector<2>, double, move_type, measure_type, double,
                  kf::policies<kf::reuse_sigma_points_t, kf::symmetrize_t>>>>);
static_assert(std::same_as<reuse_type,
                           decltype(declare(reuse_sigma_points, symmetrize))>);
static_assert(std::same_as<default_type::state, vector<2>>);
static_assert(std::same_as<default_type::output, double>);
static_assert(
    std::same_as<default_type::estimate_uncertainty, matrix<double, 2, 2>>);
static_assert(sizeof(default_type) < sizeof(reuse_type));
} // namespace
} // namespace fcarouge::test
