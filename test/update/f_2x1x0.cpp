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
#include <cmath>

namespace fcarouge::test {
namespace {
template <auto Size> using vector = column_vector<double, Size>;

//! @test Verifies the update of the multi-dimension filter declared with a
//! state transition, without process uncertainty, and with an output type
//! different from its state type, observing its first state, against its
//! equations: the declared output uncertainty participates in the innovation
//! uncertainty.
[[maybe_unused]] const auto test{[] -> int {
  const auto near{[](const double &value, const double &expected) -> bool {
    return std::abs(value - expected) < 1e-12;
  }};

  kalman filter{state{vector<2>{1., 2.}}, output<double>,
                estimate_uncertainty{{3., 0.}, {0., 1.}},
                output_uncertainty{1.}, state_transition{{1., 1.}, {0., 1.}}};

  filter.update(3.);

  assert(filter.z() == 3.);
  assert(filter.r() == 1.);
  assert(near(filter.s(), 4.));
  assert(near(filter.y(), 2.));
  assert(near(filter.k()[0], 0.75) && near(filter.k()[1], 0.));
  assert(near(filter.x()[0], 2.5) && near(filter.x()[1], 2.));
  assert(near(filter.p()(0, 0), 0.75) && near(filter.p()(0, 1), 0.));
  assert(near(filter.p()(1, 0), 0.) && near(filter.p()(1, 1), 1.));

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
