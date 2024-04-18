/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.3.0
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
//! @test Verifies the observation transition matrix H management overloads for
//! supported filter types.
[[maybe_unused]] auto test{[] {
  kalman filter;
  using kalman = decltype(filter);

  assert(filter.h() == 1);

  {
    const auto h{2.};
    filter.h(h);
    assert(filter.h() == 2);
  }

  {
    const auto h{3.};
    filter.h(h);
    assert(filter.h() == 3);
  }

  {
    const auto h{
        []([[maybe_unused]] const kalman::state &x) -> kalman::output_model {
          return 4.;
        }};
    filter.h(h);
    assert(filter.h() == 3);
    filter.update(0.);
    assert(filter.h() == 4);
  }

  {
    const auto h{
        []([[maybe_unused]] const kalman::state &x) -> kalman::output_model {
          return 5.;
        }};
    filter.h(std::move(h));
    assert(filter.h() == 4);
    filter.update(0.);
    assert(filter.h() == 5);
  }

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
