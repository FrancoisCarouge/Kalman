/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.4.0
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
#include "fcarouge/indexed.hpp"
#include "fcarouge/unit.hpp"

namespace fcarouge {

//! @name Types
//! @{

template <auto Reference, auto OriginPoint, typename Representation>
struct indexed::element_traits<
    Representation,
    mp_units::quantity_point<Reference, OriginPoint, Representation>> {
  [[nodiscard]] static inline constexpr Representation to_underlying(
      const mp_units::quantity_point<Reference, OriginPoint, Representation>
          &value) {
    return value.quantity_from(OriginPoint).numerical_value_in(value.unit);
  }

  [[nodiscard]] static inline constexpr mp_units::quantity_point<
      Reference, OriginPoint, Representation> &
  from_underlying(Representation &value) {
    return reinterpret_cast<
        mp_units::quantity_point<Reference, OriginPoint, Representation> &>(
        value);
  }
};

template <auto Reference, typename Representation>
struct indexed::element_traits<Representation,
                               mp_units::quantity<Reference, Representation>> {
  [[nodiscard]] static inline constexpr Representation
  to_underlying(const mp_units::quantity<Reference, Representation> &value) {
    return value.numerical_value_in(value.unit);
  }

  [[nodiscard]] static inline constexpr mp_units::quantity<Reference,
                                                           Representation> &
  from_underlying(Representation &value) {
    return reinterpret_cast<mp_units::quantity<Reference, Representation> &>(
        value);
  }
};

//! @brief Quantity column vector with mp-units and Eigen implementations.
template <typename Representation, typename... Types>
using column_vector = indexed::column_vector<
    eigen::column_vector<Representation, sizeof...(Types)>, Types...>;

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

} // namespace fcarouge

#endif // FCAROUGE_QUANTITY_HPP
