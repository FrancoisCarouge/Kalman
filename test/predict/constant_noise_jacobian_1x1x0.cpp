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
//! @test Verifies the extended filter with a constant callable process
//! uncertainty agrees with the extended filter whose process uncertainty is
//! declared by the caller.
[[maybe_unused]] const auto test{[] -> int {
  kalman general{
      state{3.},
      output<double>,
      estimate_uncertainty{2.},
      process_uncertainty{
          []([[maybe_unused]] const double &x,
             [[maybe_unused]] const double &step) -> double { return 0.5; }},
      output_uncertainty{1.},
      output_model{[]([[maybe_unused]] const double &x,
                      const double &scale) -> double { return scale; }},
      state_transition{[](const double &x, const double &step) -> double {
        return step * x;
      }},
      transition{[](const double &x, const double &step) -> double {
        return step * x * x / 2.;
      }},
      observation{[](const double &x, const double &scale) -> double {
        return scale * x;
      }},
      update_types<double>,
      prediction_types<double>};
  kalman specialized{
      state{3.},
      output<double>,
      estimate_uncertainty{2.},
      process_uncertainty{0.5},
      output_uncertainty{1.},
      output_model{[]([[maybe_unused]] const double &x,
                      const double &scale) -> double { return scale; }},
      state_transition{[](const double &x, const double &step) -> double {
        return step * x;
      }},
      transition{[](const double &x, const double &step) -> double {
        return step * x * x / 2.;
      }},
      observation{[](const double &x, const double &scale) -> double {
        return scale * x;
      }},
      update_types<double>,
      prediction_types<double>};

  general.predict(0.5);
  specialized.predict(0.5);
  general.update(2., 6.);
  specialized.update(2., 6.);
  general.predict(0.25);
  specialized.predict(0.25);

  assert(general.q() == specialized.q());
  assert(general.f() == specialized.f());
  assert(general.x() == specialized.x());
  assert(general.p() == specialized.p());
  assert(general.k() == specialized.k());

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
