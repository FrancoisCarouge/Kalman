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

//! @test Verifies the localization of the default Joseph update of a scalar
//! output, evaluated in two steps, applies the taper once, to the updated
//! estimate uncertainty: P = ρ ∘ ((I - KH)P(I - KH)ᵀ + KRKᵀ).
[[maybe_unused]] const auto test{[] -> int {
  const auto near{[](const double &value, const double &expected) -> bool {
    return std::abs(value - expected) < 1e-12;
  }};

  kalman filter{
      state{vector<3>{0., 0., 0.}},
      output<double>,
      estimate_uncertainty{sparse::make_matrix<covariance>(4., 2., 0_o, 2., 4.,
                                                           2., 0_o, 2., 4.)},
      process_uncertainty{{0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}},
      output_uncertainty{4.},
      output_model{[]([[maybe_unused]] const vector<3> &x) -> matrix<1, 3> {
        return matrix<1, 3>{{1., 0., 0.}};
      }},
      state_transition{[]([[maybe_unused]] const vector<3> &x,
                          [[maybe_unused]] const double &step) -> matrix<3, 3> {
        return matrix<3, 3>{{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}};
      }},
      transition{[](const vector<3> &x, [[maybe_unused]] const double &step)
                     -> vector<3> { return x; }},
      observation{[](const vector<3> &x) -> double { return x(0); }},
      prediction_types<double>,
      localize<neighbours>};

  filter.update(2.);

  assert(near(filter.x()(0), 1.));
  assert(near(filter.x()(1), 0.5));
  assert(near(filter.p().at<0, 0>(), 2.));
  assert(near(filter.p().at<0, 1>(), 0.5));
  assert(near(filter.p().at<1, 0>(), 0.5));
  assert(near(filter.p().at<1, 1>(), 3.5));
  assert(near(filter.p().at<1, 2>(), 1.));
  assert(near(filter.p().at<2, 2>(), 4.));

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
