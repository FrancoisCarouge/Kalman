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
#include <concepts>

namespace fcarouge::test {
namespace {
using literals::operator""_o;

template <auto Size> using vector = column_vector<double, Size>;
template <auto Row, auto Column> using matrix = matrix<double, Row, Column>;
using transition_jacobian =
    typed_sparse_matrix<double, sparse::upper_triangular<2>>;
using output_jacobian = typed_sparse_matrix<double, sparse::banded<1, 2, 0, 0>>;

//! @test Verifies the multi-dimension extended filter with a callable state
//! transition of the prediction arguments keeps the types its callable state
//! transition and output model compute, here of sparsity patterns, rather than
//! the dense quotients of its state and output, and predicts as with dense
//! ones: X = F.X and P = F.P.Fᵀ + Q.
[[maybe_unused]] const auto test{[] -> int {
  kalman filter{
      state{vector<2>{1., 2.}},
      output<double>,
      estimate_uncertainty{{4., 0.}, {0., 1.}},
      process_uncertainty{{0., 0.}, {0., 0.}},
      output_uncertainty{1.},
      output_model{[]([[maybe_unused]] const vector<2> &x) -> output_jacobian {
        return sparse::make_matrix<output_jacobian>(1., 0_o);
      }},
      state_transition{[](const double &step) -> transition_jacobian {
        return sparse::make_matrix<transition_jacobian>(1., step, 0_o, 1.);
      }},
      observation{[](const vector<2> &x) -> double { return x(0); }},
      prediction_types<double>};

  static_assert(
      std::same_as<decltype(filter)::state_transition, transition_jacobian>);
  static_assert(std::same_as<decltype(filter)::output_model, output_jacobian>);

  filter.predict(0.5);

  assert((filter.f() ==
          sparse::make_matrix<transition_jacobian>(1., 0.5, 0_o, 1.)));
  assert((filter.x() == vector<2>{2., 2.}));
  assert((filter.p() == matrix<2, 2>{{4.25, 0.5}, {0.5, 1.}}));

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
