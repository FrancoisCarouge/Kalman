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
template <auto Size> using vector = column_vector<double, Size>;

//! @test Verifies the default Joseph form of a scalar output and a matrix
//! state, associated at a quadratic cost, against the Joseph product.
[[maybe_unused]] const auto test{[] -> int {
  const auto near{[](const double &value, const double &expected) -> bool {
    return std::abs(value - expected) < 1e-12;
  }};

  kalman filter{
      state{vector<3>{0., 0., 0.}},
      output<double>,
      estimate_uncertainty{{2., 0.5, 0.25}, {0.5, 3., 0.5}, {0.25, 0.5, 4.}},
      process_uncertainty{{0.25, 0., 0.}, {0., 0.25, 0.}, {0., 0., 0.25}},
      output_uncertainty{1.5},
      output_model{{1., 0.5, 0.25}},
      state_transition{{1., 1., 0.}, {0., 1., 1.}, {0., 0., 1.}}};

  using uncertainty = decltype(filter)::estimate_uncertainty;

  const uncertainty prior{filter.p()};

  filter.update(2.);

  const uncertainty i{kf::make_one<uncertainty>()};
  const uncertainty expected{(i - filter.k() * filter.h()) * prior *
                                 kf::t(i - filter.k() * filter.h()) +
                             filter.k() * filter.r() * kf::t(filter.k())};

  for (int row{0}; row < 3; ++row) {
    for (int column{0}; column < 3; ++column) {
      assert(near(filter.p()(row, column), expected(row, column)));
    }
  }

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
