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
#include "fcarouge/quantity_linalg.hpp"
#include "fcarouge/unit.hpp"

#include <cassert>
#include <format>

namespace fcarouge::test {
namespace {
using state =
    fcarouge::state<quantity_vector<m, m / s, m / s2, m, m / s, m / s2>>;
using output_t = quantity_vector<m, m>;
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
inline auto output{fcarouge::output<quantity_vector<Reference...>>};

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
        auto value{identity<process_uncertainty::type>};
        value[0, 0] = 0.25 * m2;
        value[0, 1] = 0.5 * m2 / s;
        value[0, 2] = 0.5 * m2 / s2;
        value[1, 0] = 0.5 * m2 / s;
        value[1, 2] = 1 * m2 / s3;
        value[2, 0] = 0.5 * m2 / s2;
        value[2, 1] = 1 * m2 / s4;
        value[3, 3] = 0.25 * m2;
        value[3, 4] = 0.5 * m2 / s;
        value[3, 5] = 0.5 * m2 / s2;
        value[4, 3] = 0.5 * m2 / s;
        value[4, 5] = 1 * m2 / s3;
        value[5, 3] = 0.5 * m2 / s2;
        value[5, 4] = 1 * m2 / s4;
        return 0.2 * 0.2 * value;
      }()},
      output_uncertainty{{9. * m2, 0. * m2}, {0. * m2, 9. * m2}},
      output_model{[]() {
        auto value{zero<output_model::type>};
        value[0, 0] = 1. * m2;
        value[1, 3] = 1. * m2;
        return value;
      }()},
      state_transition{[]() {
        auto value{identity<state_transition::type>};
        value[0, 1] = 1. * m2 / s;
        value[0, 2] = 0.5 * m2 / s2;
        value[1, 2] = 1. * m2 / s3;
        value[3, 4] = 1. * m2 / s;
        value[3, 5] = 0.5 * m2 / s2;
        value[4, 5] = 1. * m2 / s3;
        return value;
      }()}};

  filter.predict();
  filter.update(-393.66 * m, 300.4 * m);
  filter.predict();
  filter.update(-375.93 * m, 301.78 * m);
  filter.predict();

  // Verify the example estimated state at 0.1% accuracy.
  assert(abs(1 - filter.x<0>() / (-277.8 * m)) < 0.001 &&
         abs(1 - filter.x<1>() / (148.3 * m / s)) < 0.001 &&
         abs(1 - filter.x<2>() / (94.5 * m / s2)) < 0.001 &&
         abs(1 - filter.x<3>() / (249.8 * m)) < 0.001 &&
         abs(1 - filter.x<4>() / (-85.9 * m / s)) < 0.001 &&
         abs(1 - filter.x<5>() / (-63.62 * m / s2)) < 0.001 &&
         "The state estimates expected at 0.1% accuracy.");
  //! @todo Support floating point formatting precission to make this
  //! demonstrator readable.
  assert(
      std::format("{}", filter) ==
      R"({"f": [[1 m², 1 m²/s, 0.5 m²/s², 0 m², 0 m²/s, 0 m²/s²], [0 m²/s, 1 m²/s², 1 m²/s³, 0 m²/s, 0 m²/s², 0 m²/s³], [0 m²/s², 0 m²/s³, 1 m²/s⁴, 0 m²/s², 0 m²/s³, 0 m²/s⁴], [0 m², 0 m²/s, 0 m²/s², 1 m², 1 m²/s, 0.5 m²/s²], [0 m²/s, 0 m²/s², 0 m²/s³, 0 m²/s, 1 m²/s², 1 m²/s³], [0 m²/s², 0 m²/s³, 0 m²/s⁴, 0 m²/s², 0 m²/s³, 1 m²/s⁴]],)"
      R"( "h": [[1 m², 0 m²/s, 0 m²/s², 0 m², 0 m²/s, 0 m²/s²], [0 m², 0 m²/s, 0 m²/s², 1 m², 0 m²/s, 0 m²/s²]],)"
      R"( "k": [[0.99083244599683 m², 0 m²], [1.2594398947629197 m²/s, 0 m²/s], [0.5695523102683997 m²/s², 0 m²/s²], [0 m², 0.99083244599683 m²], [0 m²/s, 1.2594398947629197 m²/s], [0 m²/s², 0.5695523102683997 m²/s²]],)"
      R"( "p": [[204.88241125483438 m², 253.9791279194632 m²/s, 143.82430625534872 m²/s², 0 m², 0 m²/s, 0 m²/s²], [253.9791279194631 m²/s, 338.50137054025123 m²/s², 201.96634493213992 m²/s³, 0 m²/s, 0 m²/s², 0 m²/s³], [143.82430625534852 m²/s², 201.96634493213975 m²/s³, 126.53601893841366 m²/s⁴, 0 m²/s², 0 m²/s³, 0 m²/s⁴], [0 m², 0 m²/s, 0 m²/s², 204.88241125483438 m², 253.9791279194632 m²/s, 143.82430625534872 m²/s²], [0 m²/s, 0 m²/s², 0 m²/s³, 253.9791279194631 m²/s, 338.50137054025123 m²/s², 201.96634493213992 m²/s³], [0 m²/s², 0 m²/s³, 0 m²/s⁴, 143.82430625534852 m²/s², 201.96634493213975 m²/s³, 126.53601893841366 m²/s⁴]],)"
      R"( "q": [[0.010000000000000002 m², 0.020000000000000004 m²/s, 0.020000000000000004 m²/s², 0 m², 0 m²/s, 0 m²/s²], [0.020000000000000004 m²/s, 0.04000000000000001 m²/s², 0.04000000000000001 m²/s³, 0 m²/s, 0 m²/s², 0 m²/s³], [0.020000000000000004 m²/s², 0.04000000000000001 m²/s³, 0.04000000000000001 m²/s⁴, 0 m²/s², 0 m²/s³, 0 m²/s⁴], [0 m², 0 m²/s, 0 m²/s², 0.010000000000000002 m², 0.020000000000000004 m²/s, 0.020000000000000004 m²/s²], [0 m²/s, 0 m²/s², 0 m²/s³, 0.020000000000000004 m²/s, 0.04000000000000001 m²/s², 0.04000000000000001 m²/s³], [0 m²/s², 0 m²/s³, 0 m²/s⁴, 0.020000000000000004 m²/s², 0.04000000000000001 m²/s³, 0.04000000000000001 m²/s⁴]],)"
      R"( "r": [[9 m², 0 m²], [0 m², 9 m²]],)"
      R"( "s": [[981.7231506776836 m², 0 m²], [0 m², 981.7231506776836 m²]],)"
      R"( "x": [[-277.77625017115565 m], [148.3387453628104 m/s], [94.53276232301535 m/s²], [249.76643344114504 m], [-85.9268832930232 m/s], [-63.645643576529494 m/s²]],)"
      R"( "y": [[318.3634774825619 m], [-228.03192052980125 m]],)"
      R"( "z": [[-375.93 m], [301.78 m]]})");

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
