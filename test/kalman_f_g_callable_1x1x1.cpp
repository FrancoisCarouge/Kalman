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
//! @test Verifies the state transition and control transition callables
//! management overloads with capturing callables: the characteristics are
//! replaced in place, without allocation.
[[maybe_unused]] const auto test{[] -> int {
  const auto make_transition{[](double factor) -> auto {
    return
        [factor]([[maybe_unused]] const double &u) -> double { return factor; };
  }};
  const auto make_control{[](double factor) -> auto {
    return [factor] -> double { return factor; };
  }};
  kalman filter{
      state{1.},
      output<double>,
      input<double>,
      estimate_uncertainty{1.},
      process_uncertainty{
          []([[maybe_unused]] const double &x) -> double { return 0.; }},
      output_uncertainty{1.},
      state_transition{make_transition(1.)},
      input_control{make_control(1.)},
      prediction_types<>};

  filter.f(make_transition(3.));
  filter.g(make_control(2.));
  filter.predict(1.);

  assert(filter.f() == 3.);
  assert(filter.g() == 2.);
  assert(filter.x() == 5.);

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
