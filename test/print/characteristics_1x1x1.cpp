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
#include "fcarouge/realtime.hpp"

#include <cassert>

namespace fcarouge::test {
namespace {
//! @test Verifies the printer decorator forwards the characteristics accesses,
//! the prediction, and the update to the filter it decorates.
[[maybe_unused]] const auto test{[] -> int {
  const not_realtime opt_out;
  auto filter{kalman{state{0.}, output<double>, input<double>} | print};

  filter.f(2.);
  filter.g(3.);
  filter.h(4.);
  filter.k(5.);
  filter.p(6.);
  filter.q(7.);
  filter.r(8.);
  filter.s(9.);
  filter.u(10.);
  filter.x(11.);
  filter.y(12.);
  filter.z(13.);

  assert(filter.f() == 2.);
  assert(filter.g() == 3.);
  assert(filter.h() == 4.);
  assert(filter.k() == 5.);
  assert(filter.p() == 6.);
  assert(filter.q() == 7.);
  assert(filter.r() == 8.);
  assert(filter.s() == 9.);
  assert(filter.u() == 10.);
  assert(filter.x() == 11.);
  assert(filter.y() == 12.);
  assert(filter.z() == 13.);

  filter.predict(1.);

  assert(filter.u() == 1.);
  assert(filter.x() == 25.);
  assert(filter.p() == 31.);

  filter.update(100.);

  assert(filter.z() == 100.);
  assert(filter.y() == 0.);
  assert(filter.x() == 25.);

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
