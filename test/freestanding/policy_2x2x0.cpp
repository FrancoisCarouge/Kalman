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

#include <cstddef>

namespace fcarouge::test {
namespace {
template <auto Size> using vector = column_vector<double, Size>;
template <auto Row, auto Column> using matrix = matrix<double, Row, Column>;

//! @test Verifies the policies of a multi-dimension filter on a freestanding
//! implementation: the conditioning after a prediction against its equations,
//! the update forms against the default Joseph form.
static_assert([] -> bool {
  const auto near{[](const auto &value, const auto &expected) -> bool {
    for (std::size_t position{0}; position < value.elements.size();
         ++position) {
      const double difference{value[position] - expected[position]};

      if (difference > 1e-12 || difference < -1e-12) {
        return false;
      }
    }

    return true;
  }};
  const auto declare{[](const auto &...policies) -> auto {
    return kalman{state{vector<2>{0., 0.}},
                  output<vector<2>>,
                  estimate_uncertainty{{0.25, 0.5}, {0., 1.}},
                  process_uncertainty{{0., 0.}, {0., 0.}},
                  output_uncertainty{{1., 0.}, {0., 1.}},
                  output_model{{0., 1.}, {1., 0.}},
                  state_transition{{1., 0.}, {0., 1.}},
                  policies...};
  }};

  auto conditioned{declare(floors{cw<1.>, cw<0.5>}, symmetrize)};
  auto joseph{declare(symmetrize)};
  auto simple{declare(symmetrize, simple_form)};
  auto symmetric{declare(symmetric_form, symmetrize)};
  auto information{declare(information_form, symmetrize)};

  conditioned.predict();
  joseph.predict();
  simple.predict();
  symmetric.predict();
  information.predict();
  joseph.update(1., 3.);
  simple.update(1., 3.);
  symmetric.update(1., 3.);
  information.update(1., 3.);

  return conditioned.p() == matrix<2, 2>{{1., 0.25}, {0.25, 1.}} &&
         joseph.p()[0, 1] == joseph.p()[1, 0] && near(simple.p(), joseph.p()) &&
         near(symmetric.p(), joseph.p()) && near(information.p(), joseph.p());
}());
} // namespace
} // namespace fcarouge::test
