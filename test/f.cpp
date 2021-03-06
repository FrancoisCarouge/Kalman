/*_  __          _      __  __          _   _
 | |/ /    /\   | |    |  \/  |   /\   | \ | |
 | ' /    /  \  | |    | \  / |  /  \  |  \| |
 |  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
 | . \  / ____ \| |____| |  | |/ ____ \| |\  |
 |_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter for C++
Version 0.1.0
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

#include "fcarouge/kalman_eigen.hpp"

#include <cassert>

namespace fcarouge::test
{
namespace
{
//! @test Verifies the state transition matrix F management overloads for
//! the default filter type.
[[maybe_unused]] auto f111{ [] {
  fcarouge::kalman k;
  using kalman = decltype(k);

  assert(k.f() == 1);

  {
    const auto f{ 2. };
    k.f(f);
    assert(k.f() == 2);
  }

  {
    const auto f{ 3. };
    k.f(std::move(f));
    assert(k.f() == 3);
  }

  {
    const auto f{ 4. };
    k.f(f);
    assert(k.f() == 4);
  }

  {
    const auto f{ 5. };
    k.f(std::move(f));
    assert(k.f() == 5);
  }

  {
    const auto f{ [](const kalman::state &x) -> kalman::state_transition {
      static_cast<void>(x);
      return 6.;
    } };
    k.f(f);
    assert(k.f() == 5);
    k.predict();
    assert(k.f() == 6);
  }

  {
    const auto f{ [](const kalman::state &x) -> kalman::state_transition {
      static_cast<void>(x);
      return 7.;
    } };
    k.f(std::move(f));
    assert(k.f() == 6);
    k.predict();
    assert(k.f() == 7);
  }

  return 0;
}() };

//! @test Verifies the state transition matrix F management overloads for
//! the Eigen filter type.
[[maybe_unused]] auto f543{ [] {
  using kalman =
      eigen::kalman<double, 5, 4, 3, std::tuple<double, float, int, char>,
                    std::tuple<char, int, float, double>>;

  kalman k;
  const auto i5x5{ Eigen::Matrix<double, 5, 5>::Identity() };
  const auto z5x5{ Eigen::Matrix<double, 5, 5>::Zero() };
  const Eigen::Matrix<double, 3, 1> z3x1{ Eigen::Matrix<double, 3, 1>::Zero() };

  assert(k.f() == i5x5);

  {
    const auto f{ i5x5 };
    k.f(f);
    assert(k.f() == i5x5);
  }

  {
    const auto f{ z5x5 };
    k.f(std::move(f));
    assert(k.f() == z5x5);
  }

  {
    const auto f{ i5x5 };
    k.f(f);
    assert(k.f() == i5x5);
  }

  {
    const auto f{ z5x5 };
    k.f(std::move(f));
    assert(k.f() == z5x5);
  }

  {
    const auto f{ [](const kalman::state &x, const char &c, const int &i,
                     const float &f, const double &d,
                     const kalman::input &u) -> kalman::state_transition {
      static_cast<void>(x);
      static_cast<void>(d);
      static_cast<void>(f);
      static_cast<void>(i);
      static_cast<void>(c);
      static_cast<void>(u);
      return Eigen::Matrix<double, 5, 5>::Identity();
    } };
    k.f(f);
    assert(k.f() == z5x5);
    k.predict(char(0), 0, 0.f, 0., z3x1);
    assert(k.f() == i5x5);
  }

  {
    const auto f{ [](const kalman::state &x, const char &c, const int &i,
                     const float &f, const double &d,
                     const kalman::input &u) -> kalman::state_transition {
      static_cast<void>(x);
      static_cast<void>(d);
      static_cast<void>(f);
      static_cast<void>(i);
      static_cast<void>(c);
      static_cast<void>(u);
      return Eigen::Matrix<double, 5, 5>::Zero();
    } };
    k.f(std::move(f));
    assert(k.f() == i5x5);
    k.predict(0, 0, 0.f, 0., z3x1);
    assert(k.f() == z5x5);
  }

  return 0;
}() };

} // namespace
} // namespace fcarouge::test
