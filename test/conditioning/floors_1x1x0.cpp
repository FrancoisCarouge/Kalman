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
//! @test Verifies the floors of a runtime and of a constant standard deviation
//! raise the estimate uncertainty variance to the square of the standard
//! deviation floor after an update and a prediction, and keep a variance above
//! it.
[[maybe_unused]] const auto test{[] -> int {
  kalman runtime{state{0.},
                 output<double>,
                 estimate_uncertainty{1.},
                 process_uncertainty{0.},
                 output_uncertainty{1.},
                 floors{0.75}};
  kalman constant{state{0.},
                  output<double>,
                  estimate_uncertainty{1.},
                  process_uncertainty{0.},
                  output_uncertainty{1.},
                  floors{cw<0.75>}};
  kalman low{state{0.},
             output<double>,
             estimate_uncertainty{1.},
             process_uncertainty{0.},
             output_uncertainty{1.},
             floors{0.5}};

  runtime.update(1.);
  constant.update(1.);
  low.update(1.);

  assert(runtime.p() == 0.5625);
  assert(constant.p() == 0.5625);
  assert(low.p() == 0.5);

  runtime.p(0.25);
  constant.p(0.25);
  runtime.predict();
  constant.predict();

  assert(runtime.p() == 0.5625);
  assert(constant.p() == 0.5625);

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
