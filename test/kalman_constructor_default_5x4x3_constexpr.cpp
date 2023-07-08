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

namespace fcarouge::test {
namespace {
template <auto Size> using vector = column_vector<double, Size>;

template <auto RowSize, auto ColumnSize>
using matrix = matrix<double, RowSize, ColumnSize>;

//! @test Verifies default values are initialized for multi-dimension filters.
//!
//! @todo Make the lambda `consteval` when MSVC supports it.
[[maybe_unused]] constexpr auto test{[] {
  using kalman = kalman<vector<5>, vector<4>, vector<3>>;

  kalman filter;

  constexpr auto z3x1{zero_v<vector<3>>};
  constexpr auto i4x4{identity_v<matrix<4, 4>>};
  constexpr auto i4x5{identity_v<matrix<4, 5>>};
  constexpr auto i5x3{identity_v<matrix<5, 3>>};
  constexpr auto i5x4{identity_v<matrix<5, 4>>};
  constexpr auto i5x5{identity_v<matrix<5, 5>>};
  constexpr auto z4x1{zero_v<vector<4>>};
  constexpr auto z4x4{zero_v<matrix<4, 4>>};
  constexpr auto z5x1{zero_v<vector<5>>};
  constexpr auto z5x5{zero_v<matrix<5, 5>>};

  assert(filter.f() == i5x5);
  assert(filter.g() == i5x3);
  assert(filter.h() == i4x5);
  assert(filter.k() == i5x4);
  assert(filter.p() == i5x5);
  assert(filter.q() == z5x5 && "No process noise by default.");
  assert(filter.r() == z4x4 && "No observation noise by default.");
  assert(filter.s() == i4x4);
  assert(filter.u() == z3x1 && "No initial control.");
  assert(filter.x() == z5x1 && "Origin state.");
  assert(filter.y() == z4x1);
  assert(filter.z() == z4x1);

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
