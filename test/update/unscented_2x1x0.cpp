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

//! @test Verifies the unscented filter of a linear system agrees with the
//! Kalman filter through predictions and updates with its default sigma points
//! drawn anew, disagrees with its sigma points reused from the prediction,
//! which omit the process uncertainty, and draws them anew without a
//! prediction.
[[maybe_unused]] const auto test{[] -> int {
  const auto near{[](const double &value, const double &expected) -> bool {
    return std::abs(value - expected) < 1e-9;
  }};
  const auto declare{[](const auto &...policies) -> auto {
    return kalman{
        state{vector<2>{0., 0.}},
        output<double>,
        estimate_uncertainty{{100., 0.}, {0., 10.}},
        process_uncertainty{{0.025, 0.05}, {0.05, 0.1}},
        output_uncertainty{4.},
        transition{[](const vector<2> &x) -> vector<2> {
          return vector<2>{x[0] + x[1], x[1]};
        }},
        observation{[](const vector<2> &x) -> double { return x[0]; }},
        unscented{0.5, 2., 1.},
        policies...};
  }};

  kalman linear{state{vector<2>{0., 0.}},
                output<double>,
                estimate_uncertainty{{100., 0.}, {0., 10.}},
                process_uncertainty{{0.025, 0.05}, {0.05, 0.1}},
                output_uncertainty{4.},
                output_model{{1., 0.}},
                state_transition{{1., 1.}, {0., 1.}}};
  auto redrawn{declare()};
  auto reused{declare(reuse_sigma_points)};

  for (int step{1}; step <= 10; ++step) {
    linear.predict();
    redrawn.predict();
    reused.predict();
    linear.update(static_cast<double>(step));
    redrawn.update(static_cast<double>(step));
    reused.update(static_cast<double>(step));
  }

  for (int row{0}; row < 2; ++row) {
    assert(near(redrawn.x()[row], linear.x()[row]));

    for (int column{0}; column < 2; ++column) {
      assert(near(redrawn.p()(row, column), linear.p()(row, column)));
    }
  }

  assert(reused.p()(0, 0) > 1.01 * linear.p()(0, 0) &&
         "The reused sigma points omit the process uncertainty from the "
         "innovation uncertainty and the gain.");

  auto drawn{declare()};
  auto retained{declare(reuse_sigma_points)};

  drawn.update(1.);
  retained.update(1.);
  drawn.update(2.);
  retained.update(2.);

  for (int row{0}; row < 2; ++row) {
    assert(near(retained.x()[row], drawn.x()[row]));

    for (int column{0}; column < 2; ++column) {
      assert(near(retained.p()(row, column), drawn.p()(row, column)));
    }
  }

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
