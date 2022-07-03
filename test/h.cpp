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
//! @test Verify the observation transition matrix H management overloads for
//! the default filter type.
[[maybe_unused]] auto h111{ [] {
  fcarouge::kalman k;
  using kalman = decltype(k);

  assert(k.h() == 1);

  {
    const auto h{ 2. };
    k.h(h);
    assert(k.h() == 2);
  }

  {
    const auto h{ 3. };
    k.h(std::move(h));
    assert(k.h() == 3);
  }

  {
    const auto h{ 4. };
    k.h(h);
    assert(k.h() == 4);
  }

  {
    const auto h{ 5. };
    k.h(std::move(h));
    assert(k.h() == 5);
  }

  {
    const auto h{ [](const kalman::state &x) -> kalman::output_model {
      static_cast<void>(x);
      return 6.;
    } };
    k.h(h);
    assert(k.h() == 5);
    k.update(0.);
    assert(k.h() == 6);
  }

  {
    const auto h{ [](const kalman::state &x) -> kalman::output_model {
      static_cast<void>(x);
      return 7.;
    } };
    k.h(std::move(h));
    assert(k.h() == 6);
    k.update(0.);
    assert(k.h() == 7);
  }

  return 0;
}() };

//! @test Verify the observation transition matrix H management overloads for
//! the Eigen filter type.
[[maybe_unused]] auto h543{ [] {
  using kalman =
      eigen::kalman<double, 5, 4, 3, std::tuple<double, float, int, char>,
                    std::tuple<char, int, float, double>>;

  kalman k;
  const auto i4x5{ Eigen::Matrix<double, 4, 5>::Identity() };
  const auto z4x5{ Eigen::Matrix<double, 4, 5>::Zero() };

  assert(k.h() == i4x5);

  {
    const auto h{ i4x5 };
    k.h(h);
    assert(k.h() == i4x5);
  }

  {
    const auto h{ z4x5 };
    k.h(std::move(h));
    assert(k.h() == z4x5);
  }

  {
    const auto h{ i4x5 };
    k.h(h);
    assert(k.h() == i4x5);
  }

  {
    const auto h{ z4x5 };
    k.h(std::move(h));
    assert(k.h() == z4x5);
  }

  {
    const auto h{ [](const kalman::state &x, const double &d, const float &f,
                     const int &i, const char &c) -> kalman::output_model {
      static_cast<void>(x);
      static_cast<void>(d);
      static_cast<void>(f);
      static_cast<void>(i);
      static_cast<void>(c);
      return Eigen::Matrix<double, 4, 5>::Identity();
    } };
    k.h(h);
    assert(k.h() == z4x5);
    k.update(0., 0.f, 0, 0);
    assert(k.h() == i4x5);
  }

  {
    const auto h{ [](const kalman::state &x, const double &d, const float &f,
                     const int &i, const char &c) -> kalman::output_model {
      static_cast<void>(x);
      static_cast<void>(d);
      static_cast<void>(f);
      static_cast<void>(i);
      static_cast<void>(c);
      return Eigen::Matrix<double, 4, 5>::Zero();
    } };
    k.h(std::move(h));
    assert(k.h() == i4x5);
    k.update(0., 0.f, 0, 0);
    assert(k.h() == z4x5);
  }

  return 0;
}() };

} // namespace
} // namespace fcarouge::test
