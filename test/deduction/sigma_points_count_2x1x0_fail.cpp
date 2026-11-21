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
#include "fcarouge/linalg.hpp"

namespace fcarouge::test {
namespace {
template <auto Size> using vector = column_vector<double, Size>;

constexpr auto move{[](const vector<2> &x) -> vector<2> { return x; }};
constexpr auto measure{[](const vector<2> &x) -> double { return x[0]; }};

//! @test Verifies a declaration of the explicit default sigma points drawn
//! anew and of the reused sigma points is rejected.
[[maybe_unused]] const auto test{[] -> int {
  const estimate_uncertainty initial{{1., 0.}, {0., 1.}};
  const process_uncertainty process{{0., 0.}, {0., 0.}};
  const output_uncertainty measurement{1.};

  // Intended: kalman filter{state{vector<2>{0., 0.}}, output<double>, initial,
  //                         process, measurement, transition{move},
  //                         observation{measure},
  //                         unscented{1., 2., 0.}, reuse_sigma_points};
  kalman filter{state{vector<2>{0., 0.}},
                output<double>,
                initial,
                process,
                measurement,
                transition{move},
                observation{measure},
                unscented{1., 2., 0.},
                redraw_sigma_points,
                reuse_sigma_points};

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
