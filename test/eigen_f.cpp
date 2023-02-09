/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.2.0
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

#include <Eigen/Eigen>

#include <cassert>

template <non_arithmetic Numerator, non_arithmetic Denominator>
auto operator/(const Numerator &lhs, const Denominator &rhs)
    -> fcarouge::internal::matrix<Numerator, Denominator> {
  return rhs.transpose()
      .fullPivHouseholderQr()
      .solve(lhs.transpose())
      .transpose()
      .eval();
}

namespace fcarouge::test {
namespace {

template <auto Size> using vector = Eigen::Vector<double, Size>;

template <auto Row, auto Column>
using matrix = Eigen::Matrix<double, Row, Column>;

//! @test Verifies the state transition matrix F management overloads for
//! the Eigen filter type.
[[maybe_unused]] auto f_5x4x3{[] {
  using kalman =
      kalman<vector<5>, vector<4>, vector<3>, std::tuple<double, float, int>,
             std::tuple<int, float, double>>;

  kalman filter;

  const auto i5x5{matrix<5, 5>::Identity()};
  const auto z5x5{matrix<5, 5>::Zero()};
  const vector<3> z3{vector<3>::Zero()};

  assert(filter.f() == i5x5);

  {
    const auto f{i5x5};
    filter.f(f);
    assert(filter.f() == i5x5);
  }

  {
    const auto f{z5x5};
    filter.f(std::move(f));
    assert(filter.f() == z5x5);
  }

  {
    const auto f{i5x5};
    filter.f(f);
    assert(filter.f() == i5x5);
  }

  {
    const auto f{z5x5};
    filter.f(std::move(f));
    assert(filter.f() == z5x5);
  }

  {
    const auto f{
        []([[maybe_unused]] const kalman::state &x,
           [[maybe_unused]] const kalman::input &u,
           [[maybe_unused]] const int &i, [[maybe_unused]] const float &fp,
           [[maybe_unused]] const double &d) -> kalman::state_transition {
          return matrix<5, 5>::Identity();
        }};
    filter.f(f);
    assert(filter.f() == z5x5);
    filter.predict(0, 0.f, 0., z3);
    assert(filter.f() == i5x5);
  }

  {
    const auto f{
        []([[maybe_unused]] const kalman::state &x,
           [[maybe_unused]] const kalman::input &u,
           [[maybe_unused]] const int &i, [[maybe_unused]] const float &fp,
           [[maybe_unused]] const double &d) -> kalman::state_transition {
          return matrix<5, 5>::Zero();
        }};
    filter.f(std::move(f));
    assert(filter.f() == i5x5);
    filter.predict(0, 0.f, 0., z3);
    assert(filter.f() == z5x5);
  }

  return 0;
}()};

} // namespace
} // namespace fcarouge::test
