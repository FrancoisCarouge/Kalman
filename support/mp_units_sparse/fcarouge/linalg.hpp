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

#ifndef FCAROUGE_LINALG_HPP
#define FCAROUGE_LINALG_HPP

//! @file
//! @brief Indexed-based linear algebra with mp-units with a compile-time
//! sparsity pattern implementation.

#include "fcarouge/sparse.hpp"
#include "fcarouge/typed_linear_algebra.hpp"

#include "fcarouge/kalman_mp_units.hpp"
#include "fcarouge/kalman_sparse.hpp"
#include "fcarouge/kalman_typed_linear_algebra.hpp"
#include "fcarouge/mp_units.hpp"

#include <mp-units/framework/customization_points.h>

#include <tuple>

namespace fcarouge {
//! @brief Quantity matrix with mp-units, without structural zeros.
template <typename Representation, typename RowIndexes, typename ColumnIndexes>
using matrix =
    typed_sparse_matrix<Representation,
                        sparse::dense<std::tuple_size_v<RowIndexes>,
                                      std::tuple_size_v<ColumnIndexes>>,
                        RowIndexes, ColumnIndexes>;

//! @brief Quantity column vector with mp-units, without structural zeros.
template <typename Representation, typename... Types>
using column_vector =
    typed_sparse_column_vector<Representation,
                               sparse::dense<sizeof...(Types), 1>, Types...>;

//! @brief Quantity row vector with mp-units, without structural zeros.
template <typename Representation, typename... Types>
using row_vector =
    typed_sparse_row_vector<Representation, sparse::dense<1, sizeof...(Types)>,
                            Types...>;
} // namespace fcarouge

// The fcarouge::typed_matrix does not expose the `value_type`/`element_type`
// names the library detects automatically, so map its underlying type
// explicitly.
template <typename Type, auto Pattern, typename RowIndexes,
          typename ColumnIndexes>
struct mp_units::representation_underlying_type<fcarouge::typed_matrix<
    fcarouge::sparse_matrix<Type, Pattern>, RowIndexes, ColumnIndexes>> {
  using type = Type;
};

#endif // FCAROUGE_LINALG_HPP
