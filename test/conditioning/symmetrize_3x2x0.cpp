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

#include <cassert>

namespace fcarouge::test {
namespace {
template <auto Size> using vector = column_vector<double, Size>;

//! @test Verifies the symmetrization averages the estimate uncertainty with its
//! transpose after a prediction and an update, exactly symmetric.
[[maybe_unused]] const auto test{[] -> int {
  kalman filter{
      state{vector<3>{0., 0., 0.}},
      output<vector<2>>,
      estimate_uncertainty{{2., 0.5, 0.25}, {0.75, 3., 0.5}, {0.25, 1.5, 4.}},
      process_uncertainty{{0., 0., 0.}, {0., 0., 0.}, {0., 0., 0.}},
      output_uncertainty{{1., 0.25}, {0.25, 2.}},
      output_model{{1., 0., 0.5}, {0., 1., 0.25}},
      state_transition{{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}},
      symmetrize,
      simple_form};

  filter.predict();

  assert(filter.p()(0, 0) == 2. && filter.p()(1, 1) == 3.);
  assert(filter.p()(0, 1) == 0.625 && filter.p()(1, 0) == 0.625);
  assert(filter.p()(1, 2) == 1. && filter.p()(2, 1) == 1.);

  filter.update(1., 2.);

  for (int first{0}; first < 3; ++first) {
    for (int second{0}; second < 3; ++second) {
      assert(filter.p()(first, second) == filter.p()(second, first));
    }
  }

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
