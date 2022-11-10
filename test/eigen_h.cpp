/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter for C++
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

namespace fcarouge::test {
namespace {

template <typename Type, auto Size> using vector = Eigen::Vector<Type, Size>;

template <typename Type, auto RowSize, auto ColumnSize>
using matrix = Eigen::Matrix<Type, RowSize, ColumnSize>;

struct divide final {
  template <typename Numerator, typename Denominator>
  [[nodiscard]] inline constexpr auto
  operator()(const Numerator &numerator, const Denominator &denominator) const {
    using result =
        typename Eigen::Matrix<typename std::decay_t<Numerator>::Scalar,
                               std::decay_t<Numerator>::RowsAtCompileTime,
                               std::decay_t<Denominator>::RowsAtCompileTime>;

    return result{denominator.transpose()
                      .fullPivHouseholderQr()
                      .solve(numerator.transpose())
                      .transpose()
                      .eval()};
  }
};

//! @test Verifies the observation transition matrix H management overloads for
//! the Eigen filter type.
[[maybe_unused]] auto h_5x4x3{[] {
  using update_model = update_model<vector<double, 5>, vector<double, 4>,
                                    divide, double, float, int>;
  using prediction_model =
      prediction_model<vector<double, 5>, vector<double, 3>, int, float,
                       double>;
  using kalman = kalman<update_model, prediction_model>;

  kalman filter;
  const auto i4x5{matrix<double, 4, 5>::Identity()};
  const auto z4x5{matrix<double, 4, 5>::Zero()};
  const vector<double, 4> z4{vector<double, 4>::Zero()};

  assert(filter.h() == i4x5);

  {
    const auto h{i4x5};
    filter.h(h);
    assert(filter.h() == i4x5);
  }

  {
    const auto h{z4x5};
    filter.h(std::move(h));
    assert(filter.h() == z4x5);
  }

  {
    const auto h{i4x5};
    filter.h(h);
    assert(filter.h() == i4x5);
  }

  {
    const auto h{z4x5};
    filter.h(std::move(h));
    assert(filter.h() == z4x5);
  }

  {
    const auto h{[](const kalman::state &x, const double &d, const float &f,
                    const int &i) -> kalman::output_model {
      static_cast<void>(x);
      static_cast<void>(d);
      static_cast<void>(f);
      static_cast<void>(i);
      return matrix<double, 4, 5>::Identity();
    }};
    filter.h(h);
    assert(filter.h() == z4x5);
    filter.update(0., 0.f, 0, z4);
    assert(filter.h() == i4x5);
  }

  {
    const auto h{[](const kalman::state &x, const double &d, const float &f,
                    const int &i) -> kalman::output_model {
      static_cast<void>(x);
      static_cast<void>(d);
      static_cast<void>(f);
      static_cast<void>(i);
      return matrix<double, 4, 5>::Zero();
    }};
    filter.h(std::move(h));
    assert(filter.h() == i4x5);
    filter.update(0., 0.f, 0, z4);
    assert(filter.h() == z4x5);
  }

  return 0;
}()};

} // namespace
} // namespace fcarouge::test
