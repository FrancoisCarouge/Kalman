/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.4.0
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
#include "fcarouge/quantity.hpp"
#include "fcarouge/unit.hpp"

#include <cassert>

import std;

namespace fcarouge::test {
namespace {
template <auto... References>
using vector = fcarouge::vector<double, References...>;
using state = fcarouge::state<vector<m, m / s, m / s2, m, m / s, m / s2>>;
using output_t = vector<m, m>;
using estimate_uncertainty =
    fcarouge::estimate_uncertainty<ᴀʙᵀ<state::type, state::type>>;
using process_uncertainty =
    fcarouge::process_uncertainty<ᴀʙᵀ<state::type, state::type>>;
using output_uncertainty =
    fcarouge::output_uncertainty<ᴀʙᵀ<output_t, output_t>>;
using output_model = fcarouge::output_model<ᴀʙᵀ<output_t, state::type>>;
using state_transition =
    fcarouge::state_transition<ᴀʙᵀ<state::type, state::type>>;

template <auto... Reference>
inline fcarouge::output_t<vector<Reference...>> output{
    fcarouge::output<vector<Reference...>>};

//! @test Verifies compatibility with the `mp-units` quantities and units
//! library for C++ in the case of an algebraic 6x2x0 filter estimating the
//! vehicle location.
//!
//! @details See the sample for details.
[[maybe_unused]] auto test{[] {
  kalman filter{
      state{0. * m, 0. * m / s, 0. * m / s2, 0. * m, 0. * m / s, 0. * m / s2},
      output<m, m>,
      estimate_uncertainty{500. * identity<estimate_uncertainty::type>},
      process_uncertainty{[]() {
        process_uncertainty::type value{identity<process_uncertainty::type>};
        value.at<0, 0>() = 0.25 * m2;
        value.at<0, 1>() = 0.5 * m2 / s;
        value.at<0, 2>() = 0.5 * m2 / s2;
        value.at<1, 0>() = 0.5 * m2 / s;
        value.at<1, 2>() = 1. * m2 / s3;
        value.at<2, 0>() = 0.5 * m2 / s2;
        value.at<2, 1>() = 1. * m2 / s3;
        value.at<3, 3>() = 0.25 * m2;
        value.at<3, 4>() = 0.5 * m2 / s;
        value.at<3, 5>() = 0.5 * m2 / s2;
        value.at<4, 3>() = 0.5 * m2 / s;
        value.at<4, 5>() = 1. * m2 / s3;
        value.at<5, 3>() = 0.5 * m2 / s2;
        value.at<5, 4>() = 1. * m2 / s3;
        return 0.2 * 0.2 * value;
      }()},
      output_uncertainty{{9. * m2, 0. * m2}, {0. * m2, 9. * m2}},
      output_model{[]() {
        output_model::type value{zero<output_model::type>};
        value.at<0, 0>() = 1. * m2;
        value.at<1, 3>() = 1. * m2;
        return value;
      }()},
      state_transition{[]() {
        state_transition::type value{identity<state_transition::type>};
        value.at<0, 1>() = 1. * m2 / s;
        value.at<0, 2>() = 0.5 * m2 / s2;
        value.at<1, 2>() = 1. * m2 / s3;
        value.at<3, 4>() = 1. * m2 / s;
        value.at<3, 5>() = 0.5 * m2 / s2;
        value.at<4, 5>() = 1. * m2 / s3;
        return value;
      }()}};

  filter.predict();
  filter.update(-393.66 * m, 300.4 * m);
  filter.predict();
  filter.update(-375.93 * m, 301.78 * m);
  filter.predict();

  assert(abs(1 - filter.x().at<0>() / (-277.8 * m)) < 0.001 &&
         abs(1 - filter.x().at<1>() / (148.3 * m / s)) < 0.001 &&
         abs(1 - filter.x().at<2>() / (94.5 * m / s2)) < 0.001 &&
         abs(1 - filter.x().at<3>() / (249.8 * m)) < 0.001 &&
         abs(1 - filter.x().at<4>() / (-85.9 * m / s)) < 0.001 &&
         abs(1 - filter.x().at<5>() / (-63.62 * m / s2)) < 0.001 &&
         "The state estimates expected at 0.1% accuracy.");

  //! @todo Add format verification.

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
