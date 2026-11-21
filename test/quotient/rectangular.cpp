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

#include "fcarouge/linalg.hpp"

#include <cassert>
#include <cmath>

namespace fcarouge::test {
namespace {
//! @test Verifies the division by a rectangular denominator, from which the
//! filter deduces the types of its characteristics, solves `X * rhs = lhs`.
[[maybe_unused]] const auto test{[] -> int {
  const auto near{[](const double &value, const double &expected) -> bool {
    return std::abs(value - expected) < 1e-12;
  }};

  // A denominator of more rows than columns: the quotient of two vectors.
  const column_vector<double, 3> numerator{1., 2., 3.};
  const column_vector<double, 2> vector{1., 1.};
  const matrix<double, 3, 2> tall{numerator / vector};
  const column_vector<double, 3> tall_product{tall * vector};

  assert(near(tall_product(0), 1.));
  assert(near(tall_product(1), 2.));
  assert(near(tall_product(2), 3.));

  // A denominator of fewer rows than columns.
  const matrix<double, 2, 2> proportional{{2., 4.}, {3., 6.}};
  const row_vector<double, 2> row{1., 2.};
  const column_vector<double, 2> wide{proportional / row};
  const matrix<double, 2, 2> wide_product{wide * row};

  assert(near(wide(0), 2.));
  assert(near(wide(1), 3.));
  assert(near(wide_product(0, 1), 4.));
  assert(near(wide_product(1, 1), 6.));

  // A scalar by a column vector.
  const column_vector<double, 4> ones{1., 1., 1., 1.};
  const row_vector<double, 4> scalar{6. / ones};
  const double scalar_product{scalar * ones};

  assert(near(scalar_product, 6.));

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
