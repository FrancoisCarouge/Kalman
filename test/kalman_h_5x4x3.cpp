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
#include "fcarouge/linalg.hpp"

#include <cassert>

template <typename Numerator, fcarouge::algebraic Denominator>
constexpr auto fcarouge::operator/([[maybe_unused]] const Numerator &lhs,
                                   [[maybe_unused]] const Denominator &rhs)
    -> fcarouge::quotient<Numerator, Denominator> {
  // Definition needed, result not relevant for this test.
  return {};
}

namespace fcarouge::test {
namespace {
template <auto Size> using vector = column_vector<double, Size>;
template <auto Row, auto Column> using matrix = matrix<double, Row, Column>;

//! @test Verifies the observation transition matrix H management overloads for
//! the Eigen filter type.
[[maybe_unused]] auto test{[] {
  using kalman =
      kalman<vector<5>, vector<4>, vector<3>, std::tuple<double, float, int>,
             std::tuple<int, float, double>>;

  kalman filter;
  const auto i4x5{identity_v<matrix<4, 5>>};
  const auto z4x5{zero_v<matrix<4, 5>>};
  const auto z4{zero_v<vector<4>>};

  assert(filter.h() == i4x5);

  {
    const auto h{z4x5};
    filter.h(h);
    assert(filter.h() == z4x5);
  }

  {
    const auto h{i4x5};
    filter.h(h);
    assert(filter.h() == i4x5);
  }

  {
    const auto h{[]([[maybe_unused]] const kalman::state &x,
                    [[maybe_unused]] const double &d,
                    [[maybe_unused]] const float &f,
                    [[maybe_unused]] const int &i) -> kalman::output_model {
      return zero_v<matrix<4, 5>>;
    }};
    filter.h(h);
    assert(filter.h() == i4x5);
    filter.update(0., 0.F, 0, z4);
    assert(filter.h() == z4x5);
  }

  {
    const auto h{[]([[maybe_unused]] const kalman::state &x,
                    [[maybe_unused]] const double &d,
                    [[maybe_unused]] const float &f,
                    [[maybe_unused]] const int &i) -> kalman::output_model {
      return identity_v<matrix<4, 5>>;
    }};
    filter.h(std::move(h));
    assert(filter.h() == z4x5);
    filter.update(0., 0.F, 0, z4);
    assert(filter.h() == i4x5);
  }

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
