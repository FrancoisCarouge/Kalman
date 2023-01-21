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
#include <linear_algebra.hpp>

namespace fcarouge::test {
namespace {

template <typename Type, auto Size>
using vector = std::math::fs_matrix<Type, Size, 1>;

//! @test Verifies default values are initialized for multi-dimension filters.
[[maybe_unused]] auto p1385_initialization_5x4x3{[] {
  using kalman =
      kalman<vector<double, 5>, vector<double, 4>, vector<double, 3>>;
  kalman filter;

  return 0;
}()};

//! @test Verifies default values are initialized for multi-dimension filters,
//! no input.
[[maybe_unused]] auto defaults54{[] {
  using kalman = kalman<vector<double, 5>, vector<double, 4>>;
  kalman filter;

  return 0;
}()};

//! @test Verifies default values are initialized for multi-dimension filters,
//! single state edge case.
[[maybe_unused]] auto defaults143{[] {
  // using kalman = kalman<double, vector<double, 4>, vector<double, 3>>;
  // kalman filter;
  using kalman =
      kalman<vector<double, 1>, vector<double, 4>, vector<double, 3>>;
  kalman filter;

  return 0;
}()};

//! @test Verifies default values are initialized for multi-dimension filters,
//! single output edge case.
[[maybe_unused]] auto defaults513{[] {
  // using kalman = kalman<vector<double, 5>, double, vector<double, 3>>;
  // kalman filter;
  using kalman =
      kalman<vector<double, 5>, vector<double, 1>, vector<double, 3>>;
  kalman filter;

  return 0;
}()};

//! @test Verifies default values are initialized for multi-dimension filters,
//! single output and input edge case.
[[maybe_unused]] auto defaults511{[] {
  // using kalman = kalman<vector<double, 5>, double, double>;
  // kalman filter;
  using kalman =
      kalman<vector<double, 5>, vector<double, 1>, vector<double, 1>>;
  kalman filter;

  return 0;
}()};

//! @test Verifies default values are initialized for multi-dimension filters,
//! single state and input edge case.
[[maybe_unused]] auto defaults141{[] {
  using kalman = kalman<double, vector<double, 4>, double>;
  kalman filter;

  return 0;
}()};

//! @test Verifies default values are initialized for multi-dimension filters.
[[maybe_unused]] auto defaults113{[] {
  // using kalman = kalman<double, double, vector<double, 3>>;
  // kalman filter;
  using kalman =
      kalman<vector<double, 1>, vector<double, 1>, vector<double, 3>>;
  kalman filter;

  return 0;
}()};

} // namespace
} // namespace fcarouge::test
