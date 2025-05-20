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

//! @test Verifies default values are initialized for multi-dimension filters,
//! single state and input edge case.
[[maybe_unused]] auto test{[] {
  const matrix<4, 4> i4x4{kalman_internal::one<matrix<4, 4>>};
  const matrix<4, 1> i4x1{kalman_internal::one<matrix<4, 1>>};
  const matrix<1, 4> i1x4{kalman_internal::one<matrix<1, 4>>};
  const vector<4> z4x1{kalman_internal::zero<vector<4>>};
  const matrix<4, 4> z4x4{kalman_internal::zero<matrix<4, 4>>};
  kalman filter{state{0.0}, output<vector<4>>, input<double>};

  assert(filter.f() == 1);
  assert(filter.g() == 1);
  assert(filter.h() == i4x1);
  assert(filter.k() == i1x4);
  assert(filter.p() == 1);
  assert(filter.q() == 0 && "No process noise by default.");
  assert(filter.r() == z4x4 && "No observation noise by default.");
  assert(filter.s() == i4x4);
  assert(filter.u() == 0 && "No initial control.");
  assert(filter.x() == 0 && "Origin state.");
  assert(filter.y() == z4x1);
  assert(filter.z() == z4x1);

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
