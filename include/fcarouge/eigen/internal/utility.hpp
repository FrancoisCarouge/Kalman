/*  __          _      __  __          _   _
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

#ifndef FCAROUGE_EIGEN_INTERNAL_UTILITY_HPP
#define FCAROUGE_EIGEN_INTERNAL_UTILITY_HPP

#include <Eigen/Eigen>

#include <concepts>
#include <functional>
#include <type_traits>

namespace fcarouge::eigen::internal {

template <typename Type>
concept arithmetic = std::integral<Type> || std::floating_point<Type>;

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

  template <typename Numerator, arithmetic Denominator>
  [[nodiscard]] inline constexpr auto
  operator()(const Numerator &numerator, const Denominator &denominator) const {
    using result =
        typename Eigen::Matrix<typename std::decay_t<Numerator>::Scalar,
                               std::decay_t<Numerator>::RowsAtCompileTime, 1>;

    //! @todo Verify the correctness this operation and others for dimensions
    //! and units.
    return result{numerator / denominator};
  }

  template <arithmetic Numerator, typename Denominator>
  [[nodiscard]] inline constexpr auto
  operator()(const Numerator &numerator, const Denominator &denominator) const {
    using result =
        typename Eigen::Matrix<std::decay_t<Numerator>, 1,
                               std::decay_t<Denominator>::RowsAtCompileTime>;

    return result{
        denominator.transpose()
            .fullPivHouseholderQr()
            .solve(Eigen::Matrix<std::decay_t<Numerator>, 1, 1>(numerator))
            .transpose()
            .eval()};
  }

  template <arithmetic Numerator, arithmetic Denominator>
  [[nodiscard]] inline constexpr auto
  operator()(const Numerator &numerator, const Denominator &denominator) const {
    return numerator / denominator;
  }
};

} // namespace fcarouge::eigen::internal

#endif // FCAROUGE_EIGEN_INTERNAL_UTILITY_HPP
