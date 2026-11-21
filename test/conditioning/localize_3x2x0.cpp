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
#include <concepts>

namespace fcarouge::test {
namespace {
using literals::operator""_o;

template <auto Size> using vector = column_vector<double, Size>;
template <auto Row, auto Column> using matrix = matrix<double, Row, Column>;
using covariance = typed_sparse_matrix<double, sparse::tridiagonal<3>>;

//! @brief A positive definite taper of the correlations of neighbouring states,
//! a compile-time constant.
struct neighbours {
  static constexpr covariance value{sparse::make_matrix<covariance>(
      1., 0.5, 0_o, 0.5, 1., 0.5, 0_o, 0.5, 1.)};
};

//! @test Verifies the localization stores the estimate uncertainty in the
//! sparsity pattern of its taper, multiplies it by the taper after a prediction
//! that fills it in, and after an update, and drops the correlations beyond the
//! taper.
[[maybe_unused]] const auto test{[] -> int {
  const auto near{[](const double &value, const double &expected) -> bool {
    return std::abs(value - expected) < 1e-12;
  }};

  kalman filter{
      state{vector<3>{0., 0., 0.}},
      output<vector<2>>,
      estimate_uncertainty{sparse::make_matrix<covariance>(4., 2., 0_o, 2., 4.,
                                                           2., 0_o, 2., 4.)},
      process_uncertainty{{1., 1., 1.}, {1., 1., 1.}, {1., 1., 1.}},
      output_uncertainty{{1., 0.}, {0., 1.}},
      output_model{[]([[maybe_unused]] const vector<3> &x) -> matrix<2, 3> {
        return matrix<2, 3>{{1., 0., 0.}, {0., 0., 1.}};
      }},
      state_transition{[]([[maybe_unused]] const vector<3> &x,
                          [[maybe_unused]] const double &step) -> matrix<3, 3> {
        return matrix<3, 3>{{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      }},
      transition{[](const vector<3> &x, [[maybe_unused]] const double &step)
                     -> vector<3> { return x; }},
      observation{[](const vector<3> &x) -> vector<2> {
        return vector<2>{x(0), x(2)};
      }},
      prediction_types<double>,
      localize<neighbours>};

  static_assert(
      std::same_as<decltype(filter)::estimate_uncertainty, covariance>);

  filter.predict(1.);

  assert((filter.p() == sparse::make_matrix<covariance>(5., 1.5, 0_o, 1.5, 5.,
                                                        1.5, 0_o, 1.5, 5.)));

  filter.update(1., 1.);

  assert(near(filter.x()(0), 5. / 6.));
  assert(near(filter.x()(1), 0.5));
  assert(near(filter.x()(2), 5. / 6.));
  assert(near(filter.p().at<0, 0>(), 5. / 6.));
  assert(near(filter.p().at<0, 1>(), 0.125));
  assert(near(filter.p().at<1, 0>(), 0.125));
  assert(near(filter.p().at<1, 1>(), 4.25));
  assert(near(filter.p().at<1, 2>(), 0.125));
  assert(near(filter.p().at<2, 1>(), 0.125));
  assert(near(filter.p().at<2, 2>(), 5. / 6.));

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
