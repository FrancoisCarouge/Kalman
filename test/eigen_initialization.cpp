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

#include <Eigen/Eigen>

#include <cassert>

namespace fcarouge::test {
namespace {

template <typename Type, auto Size> using vector = Eigen::Vector<Type, Size>;

template <typename Type, auto RowSize, auto ColumnSize>
using matrix = Eigen::Matrix<Type, RowSize, ColumnSize>;

//! @test Verifies default values are initialized for multi-dimension filters.
[[maybe_unused]] auto defaults543{[] {
  using kalman =
      kalman<vector<double, 5>, vector<double, 4>, vector<double, 3>>;
  kalman filter;

  const auto z3x1{vector<double, 3>::Zero()};
  const auto i4x4{matrix<double, 4, 4>::Identity()};
  const auto i4x5{matrix<double, 4, 5>::Identity()};
  const auto i5x3{matrix<double, 5, 3>::Identity()};
  const auto i5x4{matrix<double, 5, 4>::Identity()};
  const auto i5x5{matrix<double, 5, 5>::Identity()};
  const auto z4x1{vector<double, 4>::Zero()};
  const auto z4x4{matrix<double, 4, 4>::Zero()};
  const auto z5x1{vector<double, 5>::Zero()};
  const auto z5x5{matrix<double, 5, 5>::Zero()};

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

//! @test Verifies default values are initialized for multi-dimension filters,
//! no input.
[[maybe_unused]] auto defaults54{[] {
  using kalman = kalman<vector<double, 5>, vector<double, 4>>;
  kalman filter;

  const auto i4x4{matrix<double, 4, 4>::Identity()};
  const auto i4x5{matrix<double, 4, 5>::Identity()};
  const auto i5x4{matrix<double, 5, 4>::Identity()};
  const auto i5x5{matrix<double, 5, 5>::Identity()};
  const auto z4x1{vector<double, 4>::Zero()};
  const auto z4x4{matrix<double, 4, 4>::Zero()};
  const auto z5x1{vector<double, 5>::Zero()};
  const auto z5x5{matrix<double, 5, 5>::Zero()};

  assert(filter.f() == i5x5);
  assert(filter.h() == i4x5);
  assert(filter.k() == i5x4);
  assert(filter.p() == i5x5);
  assert(filter.q() == z5x5 && "No process noise by default.");
  assert(filter.r() == z4x4 && "No observation noise by default.");
  assert(filter.s() == i4x4);
  assert(filter.x() == z5x1 && "Origin state.");
  assert(filter.y() == z4x1);
  assert(filter.z() == z4x1);

  return 0;
}()};

//! @test Verifies default values are initialized for multi-dimension filters,
//! single state edge case.
[[maybe_unused]] auto defaults143{[] {
  using kalman = kalman<double, vector<double, 4>, vector<double, 3>>;
  kalman filter;

  const auto z3x1{vector<double, 3>::Zero()};
  const auto i4x4{matrix<double, 4, 4>::Identity()};
  const auto i4x1{matrix<double, 4, 1>::Identity()};
  const auto i1x3{matrix<double, 1, 3>::Identity()};
  const auto i1x4{matrix<double, 1, 4>::Identity()};
  const auto z4x1{vector<double, 4>::Zero()};
  const auto z4x4{matrix<double, 4, 4>::Zero()};

  assert(filter.f() == 1);
  assert(filter.g() == i1x3);
  assert(filter.h() == i4x1);
  assert(filter.k() == i1x4);
  assert(filter.p() == 1);
  assert(filter.q() == 0 && "No process noise by default.");
  assert(filter.r() == z4x4 && "No observation noise by default.");
  assert(filter.s() == i4x4);
  assert(filter.u() == z3x1 && "No initial control.");
  assert(filter.x() == 0 && "Origin state.");
  assert(filter.y() == z4x1);
  assert(filter.z() == z4x1);

  return 0;
}()};

//! @test Verifies default values are initialized for multi-dimension filters,
//! single output edge case.
[[maybe_unused]] auto defaults513{[] {
  using kalman = kalman<vector<double, 5>, double, vector<double, 3>>;
  kalman filter;

  const auto z3x1{vector<double, 3>::Zero()};
  const auto i1x5{matrix<double, 1, 5>::Identity()};
  const auto i5x3{matrix<double, 5, 3>::Identity()};
  const auto i5x1{matrix<double, 5, 1>::Identity()};
  const auto i5x5{matrix<double, 5, 5>::Identity()};
  const auto z5x1{vector<double, 5>::Zero()};
  const auto z5x5{matrix<double, 5, 5>::Zero()};

  assert(filter.f() == i5x5);
  assert(filter.g() == i5x3);
  assert(filter.h() == i1x5);
  assert(filter.k() == i5x1);
  assert(filter.p() == i5x5);
  assert(filter.q() == z5x5 && "No process noise by default.");
  assert(filter.r() == 0 && "No observation noise by default.");
  assert(filter.s() == 1);
  assert(filter.u() == z3x1 && "No initial control.");
  assert(filter.x() == z5x1 && "Origin state.");
  assert(filter.y() == 0);
  assert(filter.z() == 0);

  return 0;
}()};

//! @test Verifies default values are initialized for multi-dimension filters,
//! single input edge case.
[[maybe_unused]] auto defaults541{[] {
  using kalman = kalman<vector<double, 5>, vector<double, 4>, double>;
  kalman filter;

  const auto i4x4{matrix<double, 4, 4>::Identity()};
  const auto i4x5{matrix<double, 4, 5>::Identity()};
  const auto i5x1{matrix<double, 5, 1>::Identity()};
  const auto i5x4{matrix<double, 5, 4>::Identity()};
  const auto i5x5{matrix<double, 5, 5>::Identity()};
  const auto z4x1{vector<double, 4>::Zero()};
  const auto z4x4{matrix<double, 4, 4>::Zero()};
  const auto z5x1{vector<double, 5>::Zero()};
  const auto z5x5{matrix<double, 5, 5>::Zero()};

  assert(filter.f() == i5x5);
  assert(filter.g() == i5x1);
  assert(filter.h() == i4x5);
  assert(filter.k() == i5x4);
  assert(filter.p() == i5x5);
  assert(filter.q() == z5x5 && "No process noise by default.");
  assert(filter.r() == z4x4 && "No observation noise by default.");
  assert(filter.s() == i4x4);
  assert(filter.u() == 0 && "No initial control.");
  assert(filter.x() == z5x1 && "Origin state.");
  assert(filter.y() == z4x1);
  assert(filter.z() == z4x1);

  return 0;
}()};

//! @test Verifies default values are initialized for multi-dimension filters,
//! single output and input edge case.
[[maybe_unused]] auto defaults511{[] {
  using kalman = kalman<vector<double, 5>, double, double>;
  kalman filter;

  const auto i1x5{matrix<double, 1, 5>::Identity()};
  const auto i5x1{matrix<double, 5, 1>::Identity()};
  const auto i5x5{matrix<double, 5, 5>::Identity()};
  const auto z5x1{vector<double, 5>::Zero()};
  const auto z5x5{matrix<double, 5, 5>::Zero()};

  assert(filter.f() == i5x5);
  assert(filter.g() == i5x1);
  assert(filter.h() == i1x5);
  assert(filter.k() == i5x1);
  assert(filter.p() == i5x5);
  assert(filter.q() == z5x5 && "No process noise by default.");
  assert(filter.r() == 0 && "No observation noise by default.");
  assert(filter.s() == 1);
  assert(filter.u() == 0 && "No initial control.");
  assert(filter.x() == z5x1 && "Origin state.");
  assert(filter.y() == 0);
  assert(filter.z() == 0);

  return 0;
}()};

//! @test Verifies default values are initialized for multi-dimension filters,
//! single state and input edge case.
[[maybe_unused]] auto defaults141{[] {
  using kalman = kalman<double, vector<double, 4>, double>;
  kalman filter;

  const auto i4x4{matrix<double, 4, 4>::Identity()};
  const auto i4x1{matrix<double, 4, 1>::Identity()};
  const auto i1x4{matrix<double, 1, 4>::Identity()};
  const auto z4x1{vector<double, 4>::Zero()};
  const auto z4x4{matrix<double, 4, 4>::Zero()};

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

//! @test Verifies default values are initialized for multi-dimension filters.
[[maybe_unused]] auto defaults113{[] {
  using kalman = kalman<double, double, vector<double, 3>>;
  kalman filter;

  const auto z3x1{vector<double, 3>::Zero()};
  const auto i1x3{matrix<double, 1, 3>::Identity()};

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
