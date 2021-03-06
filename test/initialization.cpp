/*_  __          _      __  __          _   _
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
#include "fcarouge/kalman_eigen.hpp"

#include <cassert>

namespace fcarouge::test
{
namespace
{
//! @test Verifies default values are initialized for single-dimension filters
//! without input control.
[[maybe_unused]] auto defaults110{ [] {
  kalman k;

  assert(k.f() == 1);
  assert(k.h() == 1);
  assert(k.k() == 1);
  assert(k.p() == 1);
  assert(k.q() == 0 && "No process noise by default.");
  assert(k.r() == 0 && "No observation noise by default.");
  assert(k.s() == 1);
  assert(k.x() == 0 && "Origin state.");
  assert(k.y() == 0);
  assert(k.z() == 0);

  return 0;
}() };

//! @test Verifies default values are initialized for single-dimension filters
//! with input control.
[[maybe_unused]] auto defaults111{ [] {
  using kalman = fcarouge::kalman<double, double, double>;
  kalman k;

  assert(k.f() == 1);
  assert(k.g() == 1);
  assert(k.h() == 1);
  assert(k.k() == 1);
  assert(k.p() == 1);
  assert(k.q() == 0 && "No process noise by default.");
  assert(k.r() == 0 && "No observation noise by default.");
  assert(k.s() == 1);
  assert(k.u() == 0 && "No initial control.");
  assert(k.x() == 0 && "Origin state.");
  assert(k.y() == 0);
  assert(k.z() == 0);

  return 0;
}() };

//! @test Verifies default values are initialized for multi-dimension filters.
[[maybe_unused]] auto defaults543{ [] {
  using kalman = eigen::kalman<double, 5, 4, 3>;

  kalman k;
  const auto z3x1{ Eigen::Vector<double, 3>::Zero() };
  const auto i4x4{ Eigen::Matrix<double, 4, 4>::Identity() };
  const auto i4x5{ Eigen::Matrix<double, 4, 5>::Identity() };
  const auto i5x3{ Eigen::Matrix<double, 5, 3>::Identity() };
  const auto i5x4{ Eigen::Matrix<double, 5, 4>::Identity() };
  const auto i5x5{ Eigen::Matrix<double, 5, 5>::Identity() };
  const auto z4x1{ Eigen::Vector<double, 4>::Zero() };
  const auto z4x4{ Eigen::Matrix<double, 4, 4>::Zero() };
  const auto z5x1{ Eigen::Vector<double, 5>::Zero() };
  const auto z5x5{ Eigen::Matrix<double, 5, 5>::Zero() };

  assert(k.f() == i5x5);
  assert(k.g() == i5x3);
  assert(k.h() == i4x5);
  assert(k.k() == i5x4);
  assert(k.p() == i5x5);
  assert(k.q() == z5x5 && "No process noise by default.");
  assert(k.r() == z4x4 && "No observation noise by default.");
  assert(k.s() == i4x4);
  assert(k.u() == z3x1 && "No initial control.");
  assert(k.x() == z5x1 && "Origin state.");
  assert(k.y() == z4x1);
  assert(k.z() == z4x1);

  return 0;
}() };

} // namespace
} // namespace fcarouge::test
