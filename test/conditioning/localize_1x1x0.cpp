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
#include <concepts>

namespace fcarouge::test {
namespace {
//! @brief A taper of a scalar estimate uncertainty, a compile-time constant.
struct half {
  static constexpr double value{0.5};
};

//! @test Verifies the localization multiplies the scalar estimate uncertainty
//! by its taper after a prediction and after an update, the gain computed from
//! the localized estimate uncertainty.
[[maybe_unused]] const auto test{[] -> int {
  kalman filter{
      state{1.},
      output<double>,
      estimate_uncertainty{2.},
      process_uncertainty{2.},
      output_uncertainty{2.},
      output_model{
          []([[maybe_unused]] const double &x) -> double { return 1.; }},
      state_transition{
          []([[maybe_unused]] const double &x,
             [[maybe_unused]] const double &step) -> double { return 1.; }},
      transition{[](const double &x, const double &step) -> double {
        return x + step;
      }},
      observation{[](const double &x) -> double { return x; }},
      prediction_types<double>,
      symmetrize,
      localize<half>};

  static_assert(std::same_as<decltype(filter)::estimate_uncertainty, double>);

  filter.predict(1.);

  assert(filter.x() == 2.);
  assert(filter.p() == 2.);

  filter.update(4.);

  assert(filter.k() == 0.5);
  assert(filter.x() == 3.);
  assert(filter.p() == 0.5);

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
