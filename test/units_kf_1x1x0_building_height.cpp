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
#include "fcarouge/unit.hpp"

#include <cassert>
#include <format>

namespace fcarouge::test {
namespace {
//! @test Verifies compatibility with the `mp-units` quantities and units
//! library for C++ in the case of an arithmetic 1x1x0 filter estimating the
//! height of a building.
//!
//! @details See the sample for details.
[[maybe_unused]] auto test{[] {
  kalman filter{state{60. * m}, output<quantity<m, double>>,
                estimate_uncertainty{225. * m2}, output_uncertainty{25. * m2}};

  filter.update(48.54 * m);
  filter.update(47.11 * m);
  filter.update(55.01 * m);
  filter.update(55.15 * m);
  filter.update(49.89 * m);
  filter.update(40.85 * m);
  filter.update(46.72 * m);
  filter.update(50.05 * m);
  filter.update(51.27 * m);
  filter.update(49.95 * m);

  assert(abs(1 - filter.x() / (49.57 * m)) < 0.001);

  //! @todo Support floating point formatting precission to make this
  //! demonstrator readable.
  assert(std::format("{}", filter) == R"({"f": 1 m²,)"
                                      R"( "k": 0.09890109890109894,)"
                                      R"( "p": 2.4725274725274726 m²,)"
                                      R"( "r": 25 m²,)"
                                      R"( "s": 27.743902439024392 m²,)"
                                      R"( "x": 49.5698901098901 m,)"
                                      R"( "y": 0.4218292682926972 m,)"
                                      R"( "z": 49.95 m})");

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
