/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter for C++
Version 0.1.0
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

#include <cassert>

namespace fcarouge::test {
namespace {
//! @test Verifies default values are initialized for single-dimension filters
//! without input control.
[[maybe_unused]] auto defaults110{[] {
  kalman filter;

  assert(filter.f() == 1);
  assert(filter.h() == 1);
  assert(filter.k() == 1);
  assert(filter.p() == 1);
  assert(filter.q() == 0 && "No process noise by default.");
  assert(filter.r() == 0 && "No observation noise by default.");
  assert(filter.s() == 1);
  assert(filter.x() == 0 && "Origin state.");
  assert(filter.y() == 0);
  assert(filter.z() == 0);

  return 0;
}()};

//! @test Verifies default values are initialized for single-dimension filters
//! with input control.
[[maybe_unused]] auto defaults111{[] {
  using kalman = fcarouge::kalman<double, double, double>;
  kalman filter;

  assert(filter.f() == 1);
  assert(filter.g() == 1);
  assert(filter.h() == 1);
  assert(filter.k() == 1);
  assert(filter.p() == 1);
  assert(filter.q() == 0 && "No process noise by default.");
  assert(filter.r() == 0 && "No observation noise by default.");
  assert(filter.s() == 1);
  assert(filter.u() == 0 && "No initial control.");
  assert(filter.x() == 0 && "Origin state.");
  assert(filter.y() == 0);
  assert(filter.z() == 0);

  return 0;
}()};

} // namespace
} // namespace fcarouge::test
