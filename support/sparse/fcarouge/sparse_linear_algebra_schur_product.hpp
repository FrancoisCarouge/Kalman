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

#ifndef FCAROUGE_SPARSE_LINEAR_ALGEBRA_SCHUR_PRODUCT_HPP
#define FCAROUGE_SPARSE_LINEAR_ALGEBRA_SCHUR_PRODUCT_HPP

//! @file
//! @brief The Schur product of the compile-time sparsity pattern storage.
//!
//! @details A compatibility layer for an operation the Sparse Linear Algebra
//! library does not provide yet, written as its own `algorithm/` would: it
//! depends on its public interface only, and is meant to move there.

#include "fcarouge/sparse_linear_algebra.hpp"

#include <array>
#include <cstddef>

namespace fcarouge::kalman_filter::internal {
//! @brief The index of each stored element of the result in the stored
//! elements of an operand, the result's pattern a subset of the operand's.
template <sparsity_pattern auto Result, sparsity_pattern auto Operand>
inline constexpr auto indexes_in{[] {
  std::array<std::size_t, Result.positions.size()> result{};

  for (std::size_t index{0}; index < Result.count; ++index) {
    result[index] = Operand.index_of(Result.positions[index].row,
                                     Result.positions[index].column);
  }

  return result;
}()};
} // namespace fcarouge::kalman_filter::internal

namespace fcarouge {
//! @brief The Schur product, or Hadamard product, of two matrices of the same
//! shape, of any patterns: the product of their elements at each position.
//!
//! @details The result stores the intersection of the operands' patterns: a
//! structural zero of either operand is a structural zero of the product. Its
//! pattern is narrower than an operand's, which drops by design the elements
//! of that operand the other does not store. Multiplying a matrix by a taper,
//! a matrix of weights of a narrower pattern, is how a filter localizes its
//! estimate uncertainty: by the Schur product theorem, the product of two
//! positive semi-definite matrices is positive semi-definite.
//!
//! @complexity Linear in the count of stored elements of the result, one
//! multiplication each. The positions of each element in the operands are
//! computed at compile time.
template <typename Type, sparsity_pattern auto Lhs, sparsity_pattern auto Rhs>
  requires(Lhs.rows == Rhs.rows and Lhs.columns == Rhs.columns)
[[nodiscard]] constexpr auto schur_product(const sparse_matrix<Type, Lhs> &lhs,
                                           const sparse_matrix<Type, Rhs> &rhs)
    -> sparse_matrix<Type, sparse::pattern_intersection<Lhs, Rhs>> {
  static constexpr auto result_pattern{sparse::pattern_intersection<Lhs, Rhs>};
  static constexpr auto &left{
      kalman_filter::internal::indexes_in<result_pattern, Lhs>};
  static constexpr auto &right{
      kalman_filter::internal::indexes_in<result_pattern, Rhs>};

  sparse_matrix<Type, result_pattern> result;

  for (std::size_t index{0}; index < result_pattern.count; ++index) {
    result.values[index] = lhs.values[left[index]] * rhs.values[right[index]];
  }

  return result;
}
} // namespace fcarouge

#endif // FCAROUGE_SPARSE_LINEAR_ALGEBRA_SCHUR_PRODUCT_HPP
