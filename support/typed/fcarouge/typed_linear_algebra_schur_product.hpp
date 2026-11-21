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

#ifndef FCAROUGE_TYPED_LINEAR_ALGEBRA_SCHUR_PRODUCT_HPP
#define FCAROUGE_TYPED_LINEAR_ALGEBRA_SCHUR_PRODUCT_HPP

//! @file
//! @brief The Schur product of typed matrices.
//!
//! @details A compatibility layer for an operation the Typed Linear Algebra
//! library does not provide yet, written as its own algorithms would: it
//! depends on its public interface only, and is meant to move there. The
//! underlying matrices provide their own Schur product, found by argument
//! dependent lookup, as the Sparse Linear Algebra compatibility layer does.

#include "fcarouge/typed_linear_algebra.hpp"

#include <tuple>
#include <type_traits>

namespace fcarouge {
namespace kalman_filter::internal {
//! @brief Whether every index type is an arithmetic type: the elements of a
//! matrix of such row and column indexes are dimensionless.
template <typename Indexes> inline constexpr bool arithmetic_indexes{false};

template <typename... Types>
inline constexpr bool arithmetic_indexes<std::tuple<Types...>>{
    (std::is_arithmetic_v<Types> and ...)};
} // namespace kalman_filter::internal

//! @brief The Schur product, or Hadamard product, of a dimensionless matrix by
//! a typed matrix of the same shape: the product of their elements at each
//! position.
//!
//! @details The elements of the left operand, a taper or a mask, are
//! dimensionless: each element of the product is of the type of the element of
//! the right operand at its position, which the result's row and column
//! indexes keep. The product of a physically typed matrix by a matrix of
//! weights scales each element without changing its unit.
template <typename LhsMatrix, typename LhsRowIndexes, typename LhsColumnIndexes,
          typename RhsMatrix, typename RhsRowIndexes, typename RhsColumnIndexes>
  requires(kalman_filter::internal::arithmetic_indexes<LhsRowIndexes> and
           kalman_filter::internal::arithmetic_indexes<LhsColumnIndexes> and
           std::tuple_size_v<LhsRowIndexes> ==
               std::tuple_size_v<RhsRowIndexes> and
           std::tuple_size_v<LhsColumnIndexes> ==
               std::tuple_size_v<RhsColumnIndexes> and
           requires(const LhsMatrix &lhs, const RhsMatrix &rhs) {
             schur_product(lhs, rhs);
           })
[[nodiscard]] constexpr auto schur_product(
    const typed_matrix<LhsMatrix, LhsRowIndexes, LhsColumnIndexes> &lhs,
    const typed_matrix<RhsMatrix, RhsRowIndexes, RhsColumnIndexes> &rhs) {
  using matrix =
      std::remove_cvref_t<decltype(schur_product(lhs.data(), rhs.data()))>;

  return typed_matrix<matrix, RhsRowIndexes, RhsColumnIndexes>{
      schur_product(lhs.data(), rhs.data())};
}
} // namespace fcarouge

#endif // FCAROUGE_TYPED_LINEAR_ALGEBRA_SCHUR_PRODUCT_HPP
