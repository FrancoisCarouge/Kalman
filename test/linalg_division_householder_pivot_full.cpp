/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.2.0
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

template <typename Numerator, fcarouge::algebraic Denominator>
constexpr auto fcarouge::operator/(const Numerator &lhs, const Denominator &rhs)
    -> fcarouge::quotient<Numerator, Denominator> {
  return rhs.transpose()
      .fullPivHouseholderQr()
      .solve(lhs.transpose())
      .transpose();
}

namespace fcarouge::test {
namespace {
template <auto Row, auto Column> using matrix = matrix<double, Row, Column>;

//! @test Verifies the Eigen decomposition solver.
//!
//! @todo Rewrite this test as a property-based test.
[[maybe_unused]] auto test{[] {
  matrix<6, 2> a{{204.882, 0}, {253.979, 0}, {143.824, 0},
                 {0, 204.882}, {0, 253.979}, {0, 143.824}};
  matrix<2, 2> r{{213.882, 0}, {0, 213.882}};
  matrix<6, 2> q{a / r};

  assert(a.isApprox(q * r));

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
