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
//! @test Verifies the prediction is independent of the declaration: the same
//! one-state, one-output, one-input filter predicts the same state from the
//! same prior and the same input, whether or not the input type equals the
//! state type, and whether or not the uncertainties are declared.
[[maybe_unused]] const auto test{[] -> int {
  kalman same_types{state{1.},
                    output<double>,
                    input<double>,
                    estimate_uncertainty{1.},
                    process_uncertainty{0.},
                    output_uncertainty{1.}};
  kalman other_input_type{state{1.},
                          output<double>,
                          input<float>,
                          estimate_uncertainty{1.},
                          process_uncertainty{0.},
                          output_uncertainty{1.}};
  kalman no_uncertainties{state{1.}, output<double>, input<double>};

  same_types.predict(2.);
  other_input_type.predict(2.F);
  no_uncertainties.predict(2.);

  assert(same_types.x() == 3.);
  assert(other_input_type.x() == 3.);
  assert(no_uncertainties.x() == 3.);

  assert(same_types.p() == 1.);
  assert(other_input_type.p() == 1.);
  assert(no_uncertainties.p() == 1.);

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
