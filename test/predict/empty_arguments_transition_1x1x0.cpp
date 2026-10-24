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
//! @test Verifies the extended filters with a transition and an observation
//! function, without update or prediction arguments, agree with the filter
//! storing both: the argument lists are a storage specialization, not a model
//! change.
[[maybe_unused]] const auto test{[] -> int {
  kalman stored{
      state{1.},
      output<double>,
      estimate_uncertainty{2.},
      process_uncertainty{0.5},
      output_uncertainty{4.},
      output_model{[]([[maybe_unused]] const double &x,
                      const double &scale) -> double { return scale; }},
      transition{[](const double &x, const double &step) -> double {
        return x + step;
      }},
      observation{[](const double &x, const double &scale) -> double {
        return scale * x;
      }},
      update_types<double>,
      prediction_types<double>};
  kalman updated{
      state{1.},
      output<double>,
      estimate_uncertainty{2.},
      process_uncertainty{0.5},
      output_uncertainty{4.},
      output_model{[]([[maybe_unused]] const double &x,
                      const double &scale) -> double { return scale; }},
      transition{[](const double &x) -> double { return x + 0.5; }},
      observation{[](const double &x, const double &scale) -> double {
        return scale * x;
      }},
      update_types<double>};
  kalman predicted{
      state{1.},
      output<double>,
      estimate_uncertainty{2.},
      process_uncertainty{0.5},
      output_uncertainty{4.},
      output_model{
          []([[maybe_unused]] const double &x) -> double { return 2.; }},
      transition{[](const double &x, const double &step) -> double {
        return x + step;
      }},
      observation{[](const double &x) -> double { return 2. * x; }},
      prediction_types<double>};
  kalman unused{state{1.},
                output<double>,
                estimate_uncertainty{2.},
                process_uncertainty{0.5},
                output_uncertainty{4.},
                output_model{[]([[maybe_unused]] const double &x) -> double {
                  return 2.;
                }},
                transition{[](const double &x) -> double { return x + 0.5; }},
                observation{[](const double &x) -> double { return 2. * x; }}};

  stored.predict(0.5);
  updated.predict();
  predicted.predict(0.5);
  unused.predict();
  stored.update(2., 6.);
  updated.update(2., 6.);
  predicted.update(6.);
  unused.update(6.);
  stored.predict(0.5);
  updated.predict();
  predicted.predict(0.5);
  unused.predict();

  assert(updated.x() == stored.x());
  assert(updated.p() == stored.p());
  assert(updated.k() == stored.k());
  assert(predicted.x() == stored.x());
  assert(predicted.p() == stored.p());
  assert(predicted.k() == stored.k());
  assert(unused.x() == stored.x());
  assert(unused.p() == stored.p());
  assert(unused.k() == stored.k());

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
