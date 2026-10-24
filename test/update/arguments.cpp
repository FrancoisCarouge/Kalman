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
//! @test Verifies the update arguments are value-initialized, then recorded by
//! the update, in order and with their declared types.
[[maybe_unused]] const auto test{[] -> int {
  kalman filter{state{0.}, output<double>, input<double>,
                update_types<int, float, double>};

  static_assert(std::same_as<decltype(filter.update<0>()), int>);
  static_assert(std::same_as<decltype(filter.update<1>()), float>);
  static_assert(std::same_as<decltype(filter.update<2>()), double>);

  assert(filter.update<0>() == 0);
  assert(filter.update<1>() == 0.F);
  assert(filter.update<2>() == 0.);

  filter.update(1, 2.F, 3., 4.);

  assert(filter.update<0>() == 1);
  assert(filter.update<1>() == 2.F);
  assert(filter.update<2>() == 3.);
  assert(filter.z() == 4.);

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
