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
//! @test Verifies the copy construction of a filter configured with capturing
//! callable characteristics: the copy and its source evolve independently.
[[maybe_unused]] const auto test{[] -> int {
  const double process_variance{0.5};
  const double output_variance{2.};
  kalman source{
      state{1.},
      output<double>,
      estimate_uncertainty{1.},
      process_uncertainty{[process_variance]([[maybe_unused]] const double &x)
                              -> double { return process_variance; }},
      output_uncertainty{[output_variance]([[maybe_unused]] const double &x,
                                           [[maybe_unused]] const double &z)
                             -> double { return output_variance; }},
      state_transition{2.}};

  decltype(source) copy{source};

  copy.predict();
  copy.update(5.);

  assert(source.x() == 1.);
  assert(source.p() == 1.);
  assert(copy.q() == process_variance);
  assert(copy.r() == output_variance);
  assert(copy.x() != 1.);

  source.predict();

  assert(source.x() == 2.);
  assert(source.q() == process_variance);

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
