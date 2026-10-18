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

//! @test Verifies the update of the default multi-dimension filter: without
//! output noise the observed states are the output, their uncertainty vanishes,
//! the unobserved state and its uncertainty are kept.
[[maybe_unused]] const auto test{[] -> int {
  const auto near{[](const double &value, const double &expected) -> bool {
    return std::abs(value - expected) < 1e-12;
  }};

  kalman filter{state{vector<5>{0., 0., 0., 0., 5.}}, output<vector<4>>,
                input<vector<3>>};

  filter.update(1., 2., 3., 4.);

  assert((filter.z() == vector<4>{1., 2., 3., 4.}));

  for (int row{0}; row < 5; ++row) {
    assert(near(filter.x()[row], row + 1.));

    for (int column{0}; column < 5; ++column) {
      assert(near(filter.p()(row, column), row == 4 && column == 4 ? 1. : 0.));
    }
  }

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
