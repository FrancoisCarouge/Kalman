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

#ifndef FCAROUGE_INTERNAL_KALMAN_EIGEN_OPERATOR_HPP
#define FCAROUGE_INTERNAL_KALMAN_EIGEN_OPERATOR_HPP

//! @file
//! @brief Kalman operation for Eigen 3 types.
//!
//! @details Default customization point objects (CPO) for Eigen 3 types.

#include "kalman.hpp"

#include <Eigen/Eigen>

#include <type_traits>

namespace fcarouge::eigen::internal
{
//! @brief Function object for performing Eigen matrix transposition.
//!
//! @details Implemented with the Eigen linear algebra library matrices with
//! sizes fixed at compile-time.
struct transpose {
  //! @brief Returns the transpose of `value`.
  //!
  //! @param value Value to compute the transpose of.
  //!
  //! @exception May throw implementation-defined exceptions.
  [[nodiscard]] inline constexpr auto operator()(const auto &value) const
  {
    using type = std::decay_t<decltype(value)>;
    using result_type =
        typename Eigen::Matrix<typename type::Scalar, type::ColsAtCompileTime,
                               type::RowsAtCompileTime>;

    return result_type{ value.transpose() };
  }
};

//! @brief Function object for performing Eigen matrix symmetrization.
//!
//! @details Implemented with the Eigen linear algebra library matrices with
//! sizes fixed at compile-time.
struct symmetrize {
  //! @brief Returns the symmetrised `value`.
  //!
  //! @param value Value to compute the symmetry of.
  //!
  //! @exception May throw implementation-defined exceptions.
  [[nodiscard]] inline constexpr auto operator()(const auto &value) const
  {
    using result_type = std::decay_t<decltype(value)>;

    return result_type{ (value + value.transpose()) / 2 };
  }
};

//! @brief Function object for performing Eigen matrix division.
//!
//! @details Implemented with the Eigen linear algebra library matrices with
//! sizes fixed at compile-time.
struct divide {
  //! @brief Returns the quotient of `numerator` and `denominator`.
  //!
  //! @param numerator The dividend matrix of the division. N: m x n
  //! @param denominator The divisor matrix of the division. D: o x n
  //!
  //! @return The quotien matrix. Q: m x o
  //!
  //! @exception May throw implementation-defined exceptions.
  //!
  //! @todo Why compilation fails if we specify the return type in the body of
  //! the function?
  [[nodiscard]] inline constexpr auto operator()(const auto &numerator,
                                                 const auto &denominator) const
      -> typename Eigen::Matrix<
          typename std::decay_t<decltype(numerator)>::Scalar,
          std::decay_t<decltype(numerator)>::RowsAtCompileTime,
          std::decay_t<decltype(denominator)>::RowsAtCompileTime>
  {
    return denominator.transpose()
        .fullPivHouseholderQr()
        .solve(numerator.transpose())
        .transpose();
  }
};

//! @brief Function object for providing an Eigen identity matrix.
//!
//! @details Implemented with the Eigen linear algebra library matrices with
//! sizes fixed at compile-time.
//!
//! @note Could this function object template be replaced by a variable
//! template? Proposed in paper P2008R0 entitled "Enabling variable template
//! template parameters".
struct identity {
  //! @brief Returns the identity maxtrix.
  //!
  //! @tparam Type The type template parameter of the matrix.
  //!
  //! @return The identity matrix `diag(1, 1, ..., 1)`.
  //!
  //! @exception May throw implementation-defined exceptions.
  template <typename Type>
  [[nodiscard]] inline constexpr auto operator()() const
  {
    return Type::Identity();
  }
};

} // namespace fcarouge::eigen::internal

#endif // FCAROUGE_INTERNAL_KALMAN_EIGEN_OPERATOR_HPP
