/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.4.0
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
#include <format>

#include <print> //////////////////////////////////////////////////////////////

namespace fcarouge::test {
namespace {
template <auto Size> using vector = column_vector<double, Size>;

//! @test Verifies formatting multi-dimension filters with input control without
//! additional arguments.
[[maybe_unused]] auto test{[] {
  kalman filter{state{vector<1>{0.}}, output<vector<4>>, input<vector<3>>};

  std::println("{}", filter); /////////////////////////////////////////////////

  assert(std::format("{}", filter) ==
         R"({"f": 1,)"
         R"( "g": [1, 0, 0],)"
         R"( "h": [[1], [0], [0], [0]],)"
         R"( "k": [1, 0, 0, 0],)"
         R"( "p": 1,)"
         R"( "q": 0,)"
         R"( "r": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],)"
         R"( "s": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],)"
         R"( "u": [[0], [0], [0]],)"
         R"( "x": 0,)"
         R"( "y": [[0], [0], [0], [0]],)"
         R"( "z": [[0], [0], [0], [0]]})");

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
