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

#ifndef FCAROUGE_QUANTITY_HPP
#define FCAROUGE_QUANTITY_HPP

//! @file
//! @brief Indexed-based linear algebra with mp-units with Eigen
//! implementations.

#include "fcarouge/eigen.hpp"
#include "fcarouge/typed_linear_algebra.hpp"
#include "fcarouge/unit.hpp"

namespace fcarouge {
template <typename To, mp_units::Quantity From>
struct element_caster<To, From> {
  [[nodiscard]] inline constexpr To operator()(const From &value) const {
    return value.numerical_value_in(value.unit);
  }
};

template <mp_units::Quantity To, typename From>
struct element_caster<To, From> {
  [[nodiscard]] inline constexpr To operator()(const From &value) const {
    return value * To::reference;
  }
};

template <mp_units::Quantity To, typename From>
struct element_caster<To &, From &> {
  [[nodiscard]] inline constexpr To &operator()(From &value) const {
    return reinterpret_cast<To &>(value);
  }
};

template <typename To, mp_units::QuantityPoint From>
struct element_caster<To, From> {
  [[nodiscard]] inline constexpr To operator()(const From &value) const {
    return value.quantity_from_zero().numerical_value_in(value.unit);
  }
};

template <mp_units::QuantityPoint To, typename From>
struct element_caster<To, From> {
  [[nodiscard]] inline constexpr To operator()(const From &value) const {
    return {value * To::unit, mp_units::default_point_origin(To::unit)};
  }
};

template <mp_units::QuantityPoint To, typename From>
struct element_caster<To &, From &> {
  [[nodiscard]] inline constexpr To &operator()(From &value) const {
    return reinterpret_cast<To &>(value);
  }
};
} // namespace fcarouge

namespace fcarouge::typed_linear_algebra_internal {
//! @brief Multiplies specialization type for uncertainty type deduction.
template <auto Reference>
struct multiplies<mp_units::quantity_point<Reference>, int> {
  [[nodiscard]] inline constexpr auto
  operator()(const mp_units::quantity_point<Reference> &lhs,
             int rhs) const -> mp_units::quantity_point<Reference>;
};

//! @brief Multiplies specialization type for uncertainty type deduction.
template <auto Reference>
struct multiplies<int, mp_units::quantity_point<Reference>> {
  [[nodiscard]] inline constexpr auto
  operator()(int lhs, const mp_units::quantity_point<Reference> &rhs) const
      -> mp_units::quantity_point<Reference>;
};

template <auto Reference1, auto Reference2>
struct multiplies<mp_units::quantity_point<Reference1>,
                  mp_units::quantity_point<Reference2>> {
  [[nodiscard]] inline constexpr auto
  operator()(const mp_units::quantity_point<Reference1> &lhs,
             const mp_units::quantity_point<Reference2> &rhs) const
      -> mp_units::quantity<Reference1 * Reference2>;
};

template <auto Reference1, auto Reference2>
struct multiplies<mp_units::quantity<Reference1>,
                  mp_units::quantity_point<Reference2>> {
  [[nodiscard]] inline constexpr auto
  operator()(const mp_units::quantity<Reference1> &lhs,
             const mp_units::quantity_point<Reference2> &rhs) const
      -> mp_units::quantity_point<Reference1 * Reference2>;
};

template <auto Reference1, auto Reference2>
struct divides<mp_units::quantity_point<Reference1>,
               mp_units::quantity_point<Reference2>> {
  [[nodiscard]] inline constexpr auto
  operator()(const mp_units::quantity_point<Reference1> &lhs,
             const mp_units::quantity_point<Reference2> &rhs) const
      -> mp_units::quantity<Reference1 / Reference2>;
};

template <auto Reference1, auto Reference2>
struct divides<mp_units::quantity<Reference1>,
               mp_units::quantity_point<Reference2>> {
  [[nodiscard]] inline constexpr auto
  operator()(const mp_units::quantity<Reference1> &lhs,
             const mp_units::quantity_point<Reference2> &rhs) const
      -> mp_units::quantity_point<Reference1 / Reference2>;
};

template <auto Reference1, auto Reference2>
struct divides<mp_units::quantity_point<Reference1>,
               mp_units::quantity<Reference2>> {
  [[nodiscard]] inline constexpr auto
  operator()(const mp_units::quantity_point<Reference1> &lhs,
             const mp_units::quantity<Reference2> &rhs) const
      -> mp_units::quantity_point<Reference1 / Reference2>;
};

} // namespace fcarouge::typed_linear_algebra_internal

namespace fcarouge {
namespace kalman_internal {

//! @brief Specialization of the evaluation type.
//!
//! @note Implementation not needed.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
struct evaluates<typed_matrix<Matrix, RowIndexes, ColumnIndexes>> {
  [[nodiscard]] inline constexpr auto operator()() const
      -> typed_matrix<evaluate<Matrix>, RowIndexes, ColumnIndexes>;
};

//! @brief Specialization of the transposes.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
struct transposes<typed_matrix<Matrix, RowIndexes, ColumnIndexes>> {
  [[nodiscard]] inline constexpr auto operator()(
      const typed_matrix<Matrix, RowIndexes, ColumnIndexes> &value) const {
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

namespace kalman_internal {
//! @brief Multiplies specialization type for uncertainty type deduction.
template <auto Reference>
struct multiplies<mp_units::quantity_point<Reference>, int> {
  [[nodiscard]] inline constexpr auto
  operator()(const mp_units::quantity_point<Reference> &lhs,
             int rhs) const -> mp_units::quantity_point<Reference>;
};

//! @brief Multiplies specialization type for uncertainty type deduction.
template <auto Reference>
struct multiplies<int, mp_units::quantity_point<Reference>> {
  [[nodiscard]] inline constexpr auto
  operator()(int lhs, const mp_units::quantity_point<Reference> &rhs) const
      -> mp_units::quantity_point<Reference>;
};

//! @}

} // namespace kalman_internal

} // namespace fcarouge

#endif // FCAROUGE_QUANTITY_HPP
