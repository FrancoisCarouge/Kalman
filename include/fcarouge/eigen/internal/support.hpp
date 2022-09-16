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

#ifndef FCAROUGE_EIGEN_INTERNAL_SUPPORT_HPP
#define FCAROUGE_EIGEN_INTERNAL_SUPPORT_HPP

#include <Eigen/Eigen>

#include <concepts>
#include <functional>
#include <type_traits>

namespace fcarouge::eigen::internal {

template <typename Type>
concept arithmetic = std::integral<Type> || std::floating_point<Type>;

struct matrix final {
  template <typename Type>
  [[nodiscard]] inline constexpr auto operator()(const Type &value) const ->
      typename std::decay_t<Type>::PlainMatrix {
    return value;
  }

  [[nodiscard]] inline constexpr auto
  operator()(const arithmetic auto &value) const {
    using type = std::decay_t<decltype(value)>;
    return Eigen::Matrix<type, 1, 1>{value};
  }
};

struct transpose final {
  template <typename Type>
  [[nodiscard]] inline constexpr auto operator()(const Type &value) const ->
      typename Eigen::Transpose<Type>::PlainMatrix {
    return value.transpose();
  }
};

struct symmetrize final {
  //! @todo Protect overflow? Is there a better way?
  [[nodiscard]] inline constexpr auto operator()(const auto &value) const {
    return (value + value.transpose()) / 2;
  }
};

//! @todo Provide a division based on `colPivHouseholderQr()`.
//! @todo Provide a division based on `householderQr()`.
struct divide final {
  template <typename Numerator, typename Denominator>
  // Numerator [m x n] / Denominator [o x n] -> Quotient [m x o]
  using result = typename Eigen::Matrix<
      typename std::decay_t<std::invoke_result_t<matrix, Numerator>>::Scalar,
      std::decay_t<std::invoke_result_t<matrix, Numerator>>::RowsAtCompileTime,
      std::decay_t<
          std::invoke_result_t<matrix, Denominator>>::RowsAtCompileTime>;

  template <typename Numerator, typename Denominator>
  [[nodiscard]] inline constexpr auto
  operator()(const Numerator &numerator, const Denominator &denominator) const
      -> result<Numerator, Denominator> {
    const matrix to_matrix;
    return to_matrix(denominator)
        .transpose()
        .fullPivHouseholderQr()
        .solve(to_matrix(numerator).transpose())
        .transpose()
        .eval();
  }
};

//! @todo Could this function object template be a variable template as proposed
//! in paper P2008R0 entitled "Enabling variable template template parameters"?
struct identity_matrix final {
  template <typename Type>
  [[nodiscard]] inline constexpr auto operator()() const -> Type {
    return Type::Identity();
  }

  template <arithmetic Type>
  [[nodiscard]] inline constexpr auto operator()() const noexcept -> Type {
    return 1;
  }
};

} // namespace fcarouge::eigen::internal

#endif // FCAROUGE_EIGEN_INTERNAL_SUPPORT_HPP
