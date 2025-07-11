/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.3
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
template <auto Row, auto Column> using matrix = matrix<double, Row, Column>;

//! @test Verifies default values are initialized for multi-dimension filters.
[[maybe_unused]] auto test{[] {
  const vector<3> z3x1{kalman_internal::zero<vector<3>>};
  const matrix<1, 3> i1x3{kalman_internal::one<matrix<1, 3>>};
  kalman filter{state{0.0}, output<double>, input<vector<3>>};

  assert(filter.f() == 1);
  assert(filter.g() == i1x3);
  assert(filter.h() == 1);
  assert(filter.k() == 1);
  assert(filter.p() == 1);
  assert(filter.q() == 0 && "No process noise by default.");
  assert(filter.r() == 0 && "No observation noise by default.");
  assert(filter.s() == 1);
  assert(filter.u() == z3x1 && "No initial control.");
  assert(filter.x() == 0 && "Origin state.");
  assert(filter.y() == 0);
  assert(filter.z() == 0);

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
