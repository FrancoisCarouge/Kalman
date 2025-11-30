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

#ifndef FCAROUGE_LINALG_HPP
#define FCAROUGE_LINALG_HPP

//! @file
//! @brief Scalar type indexed-based linear algebra with naive implementation.

#include "fcarouge/kalman_internal/utility.hpp"
#include "fcarouge/naive.hpp"
#include "fcarouge/typed_linear_algebra.hpp"

#include <cstddef>

namespace fcarouge::kalman_internal {
//! @brief Specialization of the evaluation type.
//!
//! @note Implementation not needed.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
struct evaluates<typed_matrix<Matrix, RowIndexes, ColumnIndexes>> {
  [[nodiscard]] static constexpr auto
  operator()() -> typed_matrix<evaluate<Matrix>, RowIndexes, ColumnIndexes>;
};

//! @brief Specialization of the transposes.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
struct transposes<typed_matrix<Matrix, RowIndexes, ColumnIndexes>> {
  [[nodiscard]] static constexpr auto
  operator()(const typed_matrix<Matrix, RowIndexes, ColumnIndexes> &value) {
    return typed_matrix<evaluate<transpose<Matrix>>, ColumnIndexes, RowIndexes>{
        t(value.data())};
  }
};

//! @name Algebraic Named Values
//! @{

//! @brief The one matrix indexed specialization.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
inline typed_matrix<decltype(one<Matrix>), RowIndexes, ColumnIndexes>
    one<typed_matrix<Matrix, RowIndexes, ColumnIndexes>>{one<Matrix>};

//! @brief The zero matrix indexed specialization.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
inline typed_matrix<decltype(zero<Matrix>), RowIndexes, ColumnIndexes>
    zero<typed_matrix<Matrix, RowIndexes, ColumnIndexes>>{zero<Matrix>};

//! @}

} // namespace fcarouge::kalman_internal

namespace fcarouge {

template <typename Type = double, std::size_t Row = 1, std::size_t Column = 1>
using matrix = typed_matrix<naive::matrix<Type, Row, Column>,
                            kalman_internal::tuple_n_type<Type, Row>,
                            kalman_internal::tuple_n_type<Type, Column>>;

template <typename Type = double, std::size_t Row = 1>
using column_vector = matrix<Type, Row, 1>;

} // namespace fcarouge

#endif // FCAROUGE_LINALG_HPP
