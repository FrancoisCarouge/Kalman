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

#ifndef FCAROUGE_SPARSE_HPP
#define FCAROUGE_SPARSE_HPP

//! @file
//! @brief Typed matrices over the compile-time sparsity pattern storage of the
//! Sparse Linear Algebra library.
//!
//! @details The storage, its patterns, and its operations are those of
//! <https://github.com/FrancoisCarouge/SparseLinearAlgebra>. This plug-in only
//! names the typed matrices over it and constructs them from typed values. It
//! is included before the core, so that the core's qualified calls find its
//! norms.

#include "fcarouge/kalman_filter/internal/utility.hpp"
#include "fcarouge/sparse_linear_algebra.hpp"

#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include "fcarouge/typed_linear_algebra.hpp"

namespace fcarouge {
//! @name Types
//! @{

//! @brief Typed matrix of a compile-time sparsity pattern.
//!
//! @tparam Representation The stored element representation type.
//! @tparam Pattern The sparsity pattern, of the indexes' shape.
//! @tparam RowIndexes The row indexes, by default of the representation type.
//! @tparam ColumnIndexes The column indexes, by default of the representation
//! type.
template <
    typename Representation, sparsity_pattern auto Pattern,
    typename RowIndexes =
        kalman_filter::internal::tuple_n_type<Representation, Pattern.rows>,
    typename ColumnIndexes =
        kalman_filter::internal::tuple_n_type<Representation, Pattern.columns>>
  requires(Pattern.rows == std::tuple_size_v<RowIndexes> and
           Pattern.columns == std::tuple_size_v<ColumnIndexes>)
using typed_sparse_matrix = typed_matrix<sparse_matrix<Representation, Pattern>,
                                         RowIndexes, ColumnIndexes>;

//! @brief Typed column vector of a compile-time sparsity pattern.
template <typename Representation, sparsity_pattern auto Pattern,
          typename... Types>
  requires(Pattern.rows == sizeof...(Types) and Pattern.columns == 1)
using typed_sparse_column_vector =
    typed_column_vector<sparse_matrix<Representation, Pattern>, Types...>;

//! @brief Typed row vector of a compile-time sparsity pattern.
template <typename Representation, sparsity_pattern auto Pattern,
          typename... Types>
  requires(Pattern.rows == 1 and Pattern.columns == sizeof...(Types))
using typed_sparse_row_vector =
    typed_row_vector<sparse_matrix<Representation, Pattern>, Types...>;

//! @}
} // namespace fcarouge

namespace fcarouge::sparse {
//! @name Element Construction
//! @{

//! @brief The type of the structural zero literal, `0_o`.
using structural_zero_t = decltype(literals::operator""_o < '0'>());

//! @brief Whether a value may be given at a linear position of a typed matrix:
//! a structural zero only takes the `0_o` literal, a stored element only takes
//! an implicitly convertible value.
//!
//! @details Partially specialized variable templates rather than immediately
//! invoked templated lambdas, for the constraints' evaluation by MSVC.
template <typename Value, typename TypedMatrix, std::size_t Position,
          bool Stored = TypedMatrix::matrix::structure.stores(
              Position / TypedMatrix::columns, Position % TypedMatrix::columns)>
inline constexpr bool is_given_at{std::is_convertible_v<
    Value, tla::element_at<TypedMatrix, Position / TypedMatrix::columns,
                           Position % TypedMatrix::columns>>};

template <typename Value, typename TypedMatrix, std::size_t Position>
inline constexpr bool is_given_at<Value, TypedMatrix, Position, false>{
    std::same_as<Value, structural_zero_t>};

template <typename TypedMatrix, typename Positions, typename... Values>
inline constexpr bool are_given_at{false};

template <typename TypedMatrix, std::size_t... Positions, typename... Values>
inline constexpr bool
    are_given_at<TypedMatrix, std::index_sequence<Positions...>, Values...>{
        (is_given_at<Values, TypedMatrix, Positions> and ...)};

template <typename TypedMatrix, typename... Values>
inline constexpr bool are_given{
    are_given_at<TypedMatrix, std::index_sequence_for<Values...>, Values...>};

//! @brief A typed matrix from a value at every position, in row-major order.
//!
//! @details Each position is known at compile time, and the type of each value
//! says what it is: a value for a structural zero, even a zero, or
//! `0_o` for a stored element, does not compile. A run-time value
//! needs no run-time check.
//!
//! @tparam TypedMatrix The typed matrix over a sparse storage to construct.
template <typename TypedMatrix, typename... Values>
  requires(sizeof...(Values) == TypedMatrix::rows * TypedMatrix::columns) and
          are_given<TypedMatrix, Values...>
[[nodiscard]] constexpr auto make_matrix(const Values &...values)
    -> TypedMatrix {
  TypedMatrix result;

  [&result, &values...]<std::size_t... Positions>(
      std::index_sequence<Positions...>) -> void {
    const auto write{[&result]<std::size_t Position>(
                         const auto &value) -> void {
      if constexpr (not std::same_as<std::remove_cvref_t<decltype(value)>,
                                     structural_zero_t>) {
        static constexpr std::size_t row{Position / TypedMatrix::columns};
        static constexpr std::size_t column{Position % TypedMatrix::columns};
        static constexpr std::size_t index{
            TypedMatrix::matrix::structure.index_of(row, column)};

        using element = tla::element_at<TypedMatrix, row, column>;

        // Not braces: a conversion to the element type may narrow.
        const element converted(value);

        result.data().values[index] =
            cast<typename TypedMatrix::underlying, element>(converted);
      }
    }};

    (write.template operator()<Positions>(values), ...);
  }(std::index_sequence_for<Values...>{});

  return result;
}

//! @}
} // namespace fcarouge::sparse

#endif // FCAROUGE_SPARSE_HPP
