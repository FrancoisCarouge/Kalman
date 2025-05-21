/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.3
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

#ifndef FCAROUGE_EIGEN_HPP
#define FCAROUGE_EIGEN_HPP

//! @file
//! @brief Linear algebra facade for Eigen3 third party implementation.
//!
//! @details Supporting matrix, vectors, and named algebraic values.
//!
//! @note The Eigen3 linear algebra is not constexpr-compatible as of July 2023.

#include "fcarouge/kalman_internal/utility.hpp"

#include <format>
#include <sstream>

#include <Eigen/Eigen>

namespace fcarouge::eigen {
//! @name Concepts
//! @{

//! @brief An Eigen3 algebraic concept.
template <typename Type>
concept is_eigen = requires { typename Type::PlainMatrix; };

//! @}

//! @name Types
//! @{

//! @brief Compile-time sized Eigen3 matrix.
//!
//! @details Facade for Eigen3 implementation compatibility.
//!
//! @tparam Type The matrix element type.
//! @tparam Row The number of rows of the matrix.
//! @tparam Column The number of columns of the matrix.
template <typename Type = double, auto Row = 1, auto Column = 1>
using matrix = Eigen::Matrix<Type, Row, Column>;

//! @brief Compile-time sized Eigen3 row vector.
template <typename Type = double, auto Column = 1>
using row_vector = Eigen::RowVector<Type, Column>;

//! @brief Compile-time sized Eigen3 column vector.
template <typename Type = double, auto Row = 1>
using column_vector = Eigen::Vector<Type, Row>;

//! @}

} // namespace fcarouge::eigen

namespace fcarouge::kalman_internal {
//! @brief Specialization of the evaluation type.
template <eigen::is_eigen Type> struct evaluates<Type> {
  [[nodiscard]] inline constexpr auto operator()() const ->
      typename Type::PlainMatrix;
};
} // namespace fcarouge::kalman_internal

namespace Eigen {
//! @brief Eigen matrix solution to division.
//!
//! @details Argument-dependent lookup (ADL) used for type definition orgering
//! dependencies. This demonstrator uses a householder rank-revealing QR
//! decomposition of a matrix with full pivoting. Other applications could
//! select a different solver.
template <fcarouge::eigen::is_eigen Numerator,
          fcarouge::eigen::is_eigen Denominator>
constexpr auto operator/(const Numerator &lhs, const Denominator &rhs)
    -> fcarouge::eigen::matrix<typename Denominator::Scalar,
                               Numerator::RowsAtCompileTime,
                               Denominator::RowsAtCompileTime> {
  return rhs.transpose()
      .fullPivHouseholderQr()
      .solve(lhs.transpose())
      .transpose();
}
} // namespace Eigen

//! @brief Specialization of the standard formatter for the Eigen matrix.
template <typename Type, auto Row, auto Column, typename Char>
struct std::formatter<fcarouge::eigen::matrix<Type, Row, Column>, Char> {
  constexpr auto parse(std::basic_format_parse_context<Char> &parse_context) {
    return parse_context.begin();
  }

  template <typename OutputIterator>
  constexpr auto
  format(const fcarouge::eigen::matrix<Type, Row, Column> &value,
         std::basic_format_context<OutputIterator, Char> &format_context) const
      -> OutputIterator {
    const Eigen::IOFormat output_format{Eigen::StreamPrecision,
                                        Eigen::DontAlignCols,
                                        ", ",
                                        ", ",
                                        "[",
                                        "]",
                                        "",
                                        "",
                                        ' '};

    return std::format_to(
        format_context.out(), "[{}]",
        (std::stringstream{} << value.format(output_format)).str());
  }

  template <typename OutputIterator>
  constexpr auto
  format(const fcarouge::eigen::matrix<Type, Row, Column> &value,
         std::basic_format_context<OutputIterator, Char> &format_context) const
      -> OutputIterator
    requires(fcarouge::eigen::matrix<Type, Row, Column>::RowsAtCompileTime ==
                 1 &&
             fcarouge::eigen::matrix<Type, Row, Column>::ColsAtCompileTime != 1)
  {
    const Eigen::IOFormat output_format{Eigen::StreamPrecision,
                                        Eigen::DontAlignCols,
                                        ", ",
                                        ", ",
                                        "[",
                                        "]",
                                        "",
                                        "",
                                        ' '};

    return std::format_to(
        format_context.out(), "{}",
        (std::stringstream{} << value.format(output_format)).str());
  }

  template <typename OutputIterator>
  constexpr auto
  format(const fcarouge::eigen::matrix<Type, Row, Column> &value,
         std::basic_format_context<OutputIterator, Char> &format_context) const
      -> OutputIterator
    requires(fcarouge::eigen::matrix<Type, Row, Column>::RowsAtCompileTime ==
                 1 &&
             fcarouge::eigen::matrix<Type, Row, Column>::ColsAtCompileTime == 1)
  {
    return std::format_to(format_context.out(), "{}", value.value());
  }
};

#endif // FCAROUGE_EIGEN_HPP
