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
//! @brief Indexed-based linear algebra with mp-units with Eigen
//! implementations.

#include "fcarouge/eigen.hpp"
#include "fcarouge/kalman_eigen.hpp"
#include "fcarouge/kalman_mp_units.hpp"
#include "fcarouge/kalman_typed_linear_algebra.hpp"
#include "fcarouge/typed_linear_algebra.hpp"

#include "fcarouge/mp_units.hpp"

#include <mp-units/framework/customization_points.h>
#include <mp-units/integrations/eigen.h>

#include <cstddef>
#include <tuple>

namespace fcarouge {
//! @brief Quantity matrix with mp-units and Eigen implementations.
template <typename Representation, typename RowIndexes, typename ColumnIndexes>
using matrix =
    typed_matrix<Eigen::Matrix<Representation, std::tuple_size_v<RowIndexes>,
                               std::tuple_size_v<ColumnIndexes>>,
                 RowIndexes, ColumnIndexes>;

//! @brief Quantity column vector with mp-units and Eigen implementations.
template <typename Representation, typename... Types>
using column_vector =
    typed_column_vector<eigen::column_vector<Representation, sizeof...(Types)>,
                        Types...>;

//! @brief Quantity row vector with mp-units and Eigen implementations.
template <typename Representation, typename... Types>
using row_vector =
    typed_row_vector<eigen::row_vector<Representation, sizeof...(Types)>,
                     Types...>;
} // namespace fcarouge

// Just like Eigen, the fcarouge::typed_matrix
// arithmetic operators return lazy expression templates; store their evaluated
// concrete type (`PlainObject`) in a quantity instead. Concrete
// matrices/vectors map to themselves.
//
// The `typename T::PlainObject` requirement is checked first and short-circuits
// the rest of the constraint: `representation_canonical_type` is instantiated
// for every representation type (including `int`, `double`, ...), and
// `Eigen::EigenBase<T>` is ill-formed for a non-Eigen `T`, so it must not be
// instantiated unless `T` already looks like an Eigen type.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
  requires requires { typename Matrix::PlainObject; } &&
           std::derived_from<Matrix, Eigen::EigenBase<Matrix>>
struct mp_units::representation_canonical_type<
    fcarouge::typed_matrix<Matrix, RowIndexes, ColumnIndexes>> {
  using type =
      fcarouge::typed_matrix<std::remove_cvref_t<typename Matrix::PlainObject>,
                             RowIndexes, ColumnIndexes>;
};

// Just like Blaze, the fcarouge::typed_matrix
// does not expose the `value_type`/`element_type` names the library detects
// automatically, so map its `ElementType` explicitly.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
  requires requires { typename Matrix::PlainObject; } &&
           std::derived_from<Matrix, Eigen::EigenBase<Matrix>>
struct mp_units::representation_underlying_type<
    fcarouge::typed_matrix<Matrix, RowIndexes, ColumnIndexes>> {
  using type =
      fcarouge::typed_matrix<Matrix, RowIndexes, ColumnIndexes>::underlying;
};

#endif // FCAROUGE_LINALG_HPP
