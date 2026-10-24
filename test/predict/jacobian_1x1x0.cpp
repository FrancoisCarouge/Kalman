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

#include <cassert>

namespace fcarouge::test {
namespace {
//! @test Verifies the prediction of the extended filter with a callable state
//! transition and a transition function of the state and the prediction
//! arguments: F = ∂f/∂X is computed at the state estimate before the
//! prediction, X = f(X, arguments), and P = F.P.Fᵀ + Q.
[[maybe_unused]] const auto test{[] -> int {
  kalman filter{
      state{3.},
      output<double>,
      estimate_uncertainty{2.},
      process_uncertainty{0.5},
      output_uncertainty{1.},
      output_model{
          []([[maybe_unused]] const double &x) -> double { return 1.; }},
      state_transition{[](const double &x, const double &step) -> double {
        return step * x;
      }},
      transition{[](const double &x, const double &step) -> double {
        return step * x * x / 2.;
      }},
      observation{[](const double &x) -> double { return x; }},
      prediction_types<double>};

  filter.predict(0.5);

  assert(filter.predict<0>() == 0.5);
  assert(filter.f() == 1.5);
  assert(filter.x() == 2.25);
  assert(filter.p() == 5.);

  filter.predict(0.5);

  assert(filter.f() == 1.125);
  assert(filter.x() == 1.265625);
  assert(filter.p() == 6.828125);

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
