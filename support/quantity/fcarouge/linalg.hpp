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

#ifndef FCAROUGE_QUANTITY_HPP
#define FCAROUGE_QUANTITY_HPP

//! @file
//! @brief Indexed-based linear algebra with mp-units with Eigen
//! implementations.

#include "fcarouge/eigen.hpp"
#include "fcarouge/typed_linear_algebra.hpp"
#include "fcarouge/unit.hpp"

namespace fcarouge {
// Teach the typed linear algebra library how to convert underlying scalar types
// to and from mp-units' types.
template <typename To, mp_units::Quantity From>
struct element_caster<To, From> {
  [[nodiscard]] static constexpr auto operator()(From value) -> To {
    using representation = typename std::remove_cvref_t<From>::rep;

    static_assert(std::same_as<representation, std::remove_cvref_t<To>>,
                  "The underlying storage type must be identical to the "
                  "quantity representation type to guarantee the conversion is "
                  "explicitely decided by the end-user.");

    return value.numerical_value_in(value.unit);
  }
};

template <mp_units::Quantity To, typename From>
struct element_caster<To, From> {
  [[nodiscard]] static constexpr auto operator()(From value) -> To {
    using representation = typename std::remove_cvref_t<To>::rep;

    static_assert(std::same_as<representation, std::remove_cvref_t<From>>,
                  "The underlying storage type must be identical to the "
                  "quantity representation type to guarantee the conversion is "
                  "explicitely decided by the end-user.");

    return value * To::reference;
  }
};

template <mp_units::Quantity To, typename From>
struct element_caster<To &, From &> {
  // A quantity reference cannot be safely materialized out of a representation
  // reference. It would be undefined behavior even if the size, padding,
  // alignment, aliasing are controlled. Therefore the best we can do is to
  // return a constant quantity value to inform the end-user lvalue reference
  // assignment cannot be supported.
  [[nodiscard]] static constexpr auto operator()(From value) -> const To {
    using representation = typename std::remove_cvref_t<To>::rep;

    static_assert(std::same_as<representation, std::remove_cvref_t<From>>,
                  "The underlying storage type must be identical to the "
                  "quantity representation type to guarantee the conversion is "
                  "explicitely decided by the end-user.");

    return value * To::reference;
  }
};

template <typename To, mp_units::Reference From>
struct element_caster<To, From> {
  [[nodiscard]] static constexpr auto
  operator()([[maybe_unused]] From value) -> To {
    return 1.;
  }
};
} // namespace fcarouge

namespace fcarouge {
namespace kalman_internal {

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
} // namespace kalman_internal

//! @brief Quantity column vector with mp-units and Eigen implementations.
template <typename Representation, typename... Types>
using column_vector =
    typed_column_vector<eigen::column_vector<Representation, sizeof...(Types)>,
                        Types...>;
} // namespace fcarouge

#endif // FCAROUGE_QUANTITY_HPP
