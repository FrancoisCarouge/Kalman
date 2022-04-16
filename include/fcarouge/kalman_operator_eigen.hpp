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

#ifndef FCAROUGE_KALMAN_OPERATOR_EIGEN_HPP
#define FCAROUGE_KALMAN_OPERATOR_EIGEN_HPP

//! @file
//! @brief Kalman operation for Eigen 3 types.

#include "kalman.hpp"

#include <Eigen/Eigen>

namespace fcarouge::eigen
{
template <typename Type> struct transpose {
  [[nodiscard]] inline constexpr Eigen::Matrix<
      typename Type::Scalar, Type::ColsAtCompileTime, Type::RowsAtCompileTime>
  operator()(const Type &value)
  {
    return value.transpose();
  }
};

template <typename Type> struct symmetrize {
  [[nodiscard]] inline constexpr Eigen::Matrix<
      typename Type::Scalar, Type::RowsAtCompileTime, Type::ColsAtCompileTime>
  operator()(const Type &value)
  {
    const auto e{ value.eval() };
    return (e + e.transpose()) / 2;
  }
};

template <typename Numerator, typename Denominator> struct divide {
  [[nodiscard]] inline constexpr Eigen::Matrix<typename Numerator::Scalar,
                                               Numerator::RowsAtCompileTime,
                                               Denominator::RowsAtCompileTime>
  operator()(const Numerator &numerator, const Denominator &denominator)
  {
    return denominator.transpose()
        .fullPivHouseholderQr()
        .solve(numerator.transpose())
        .transpose();
  }
};

template <typename Type> struct identity {
  [[nodiscard]] inline constexpr Eigen::Matrix<
      typename Type::Scalar, Type::RowsAtCompileTime, Type::ColsAtCompileTime>
  operator()()
  {
    return Type::Identity();
  }
};

template <typename Type, int State, int Output, int Input,
          typename... PredictionArguments>
using kalman =
    fcarouge::kalman<Eigen::Vector<Type, State>, Eigen::Vector<Type, Output>,
                     Eigen::Vector<Type, Input>, transpose, symmetrize, divide,
                     identity, PredictionArguments...>;

} // namespace fcarouge::eigen

#endif // FCAROUGE_KALMAN_OPERATOR_EIGEN_HPP
