/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.4
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

#ifndef FCAROUGE_KALMAN_SPARSE_HPP
#define FCAROUGE_KALMAN_SPARSE_HPP

//! @file
//! @brief Kalman filter adaptation of the compile-time sparsity pattern
//! storage.
//!
//! @details Specializes the filter's linear algebra customization points for
//! the sparse matrix of the Sparse Linear Algebra library: the transposition,
//! and the named algebraic values, each of the matrix's own pattern. Completes
//! its division, of a square denominator only, with the rectangular division
//! the filter's characteristic types are deduced from.

#include "fcarouge/kalman_filter/internal/utility.hpp"
#include "fcarouge/sparse_linear_algebra.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace fcarouge::kalman_filter::internal {
//! @brief Specialization of the transposes.
template <typename Type, auto Pattern>
struct transposes<sparse_matrix<Type, Pattern>> {
  [[nodiscard]] static constexpr auto
  operator()(const sparse_matrix<Type, Pattern> &value) {
    return fcarouge::transposed(value);
  }
};

//! @brief The count of positions of the main diagonal of the pattern.
template <auto Pattern>
inline constexpr std::size_t diagonal_size{
    std::min(Pattern.rows, Pattern.columns)};

//! @brief Whether the pattern stores every position of its main diagonal.
template <auto Pattern>
inline constexpr bool stores_diagonal{[] -> bool {
  for (std::size_t index{0}; index < diagonal_size<Pattern>; ++index) {
    if (not Pattern.stores(index, index)) {
      return false;
    }
  }

  return true;
}()};

//! @name Algebraic Named Values
//! @{

//! @brief The one matrix specialization.
//!
//! @details Of the matrix's own pattern, which must store its main diagonal:
//! a pattern without a diagonal position has no one matrix.
template <typename Type, auto Pattern>
  requires stores_diagonal<Pattern>
struct ones<sparse_matrix<Type, Pattern>> {
  [[nodiscard]] static constexpr auto operator()()
      -> sparse_matrix<Type, Pattern> {
    sparse_matrix<Type, Pattern> result;

    [&result]<std::size_t... Indexes>(
        [[maybe_unused]] std::index_sequence<Indexes...> indexes) -> void {
      ((result.template at<Indexes, Indexes>() = Type{1}), ...);
    }(std::make_index_sequence<diagonal_size<Pattern>>{});

    return result;
  }
};

//! @brief The zero matrix specialization.
//!
//! @details Of the matrix's own pattern, its stored elements zero.
template <typename Type, auto Pattern>
struct zeros<sparse_matrix<Type, Pattern>> {
  [[nodiscard]] static constexpr auto operator()()
      -> sparse_matrix<Type, Pattern> {
    return sparse_matrix<Type, Pattern>{};
  }
};

//! @}
} // namespace fcarouge::kalman_filter::internal

namespace fcarouge {
//! @name Algorithms
//! @{

//! @brief Division by a rectangular denominator: the solution `X` of
//! `X * rhs = lhs`.
//!
//! @details The filter deduces the types of its characteristics by dividing
//! its vectors: the state transition is the quotient of the state by the state,
//! the output model of the output by the state. The minimum norm solution
//! `lhs * (rhsᵀ * rhs)⁻¹ * rhsᵀ` for a denominator of more rows than columns,
//! a vector included, the least squares solution `lhs * rhsᵀ * (rhs * rhsᵀ)⁻¹`
//! otherwise, each exact for a denominator of full rank. Both solve through the
//! division of a square denominator, of the smaller dimension.
//!
//! @warning The denominator must be of full rank: otherwise its square divides
//! by a zero pivot.
template <typename Type, auto Lhs, auto Rhs>
  requires(Lhs.columns == Rhs.columns and Rhs.rows != Rhs.columns)
[[nodiscard]] constexpr auto operator/(const sparse_matrix<Type, Lhs> &lhs,
                                       const sparse_matrix<Type, Rhs> &rhs) {
  if constexpr (Rhs.rows > Rhs.columns) {
    return lhs / (transposed(rhs) * rhs) * transposed(rhs);
  } else {
    return lhs * transposed(rhs) / (rhs * transposed(rhs));
  }
}

//! @brief Division of a scalar by a column vector: the row vector `X` of
//! `X * rhs = lhs`.
//!
//! @details The minimum norm solution `lhs * rhsᵀ / (rhsᵀ * rhs)`, as the
//! filter deduces the type of the output model of a scalar output.
//!
//! @warning The denominator must not be zero.
template <typename Type, auto Rhs>
  requires(Rhs.columns == 1)
[[nodiscard]] constexpr auto operator/(const Type &lhs,
                                       const sparse_matrix<Type, Rhs> &rhs) {
  return lhs / dot(rhs, rhs) * transposed(rhs);
}

//! @}
} // namespace fcarouge

#endif // FCAROUGE_KALMAN_SPARSE_HPP
