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

#ifndef FCAROUGE_MP_UNITS_HPP
#define FCAROUGE_MP_UNITS_HPP

//! @file
//! @brief Quantities and units facade for mp-units third party implementation.
//!
//! @details Supporting quantities, values, and functions.

#include "fcarouge/typed_linear_algebra.hpp"

#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <mp-units/framework/customization_points.h>
#include <mp-units/framework/quantity.h>
#include <mp-units/framework/quantity_cast.h>
#include <mp-units/framework/quantity_point.h>
#include <mp-units/framework/vector_components.h>
#include <mp-units/math.h>
#include <mp-units/systems/isq/thermodynamics.h>
#include <mp-units/systems/si.h>

namespace fcarouge {
// Teach the typed linear algebra library how to convert underlying scalar types
// to and from mp-units' types. Support castable quantity specifications.
template <typename To, mp_units::Quantity From>
struct element_caster<To, From> {
  [[nodiscard]] static constexpr auto operator()(mp_units::Quantity auto value)
      -> To {
    using representation = typename std::remove_cvref_t<From>::rep;

    static_assert(std::same_as<representation, std::remove_cvref_t<To>>,
                  "The underlying storage type must be identical to the "
                  "quantity representation type to guarantee the conversion is "
                  "explicitly decided by the end-user.");

    constexpr auto reference{std::remove_cvref_t<From>::reference};
    constexpr auto specification{get_quantity_spec(reference)};
    auto specified{mp_units::quantity_cast<specification>(value)};

    return specified.numerical_value_in(specified.unit);
  }
};

template <typename To, mp_units::UnitMagnitude Magnitude>
struct element_caster<To, Magnitude> {
  [[nodiscard]] static constexpr auto operator()(Magnitude value) -> To {
    return get_value<To>(value);
  }
};

template <mp_units::Quantity To, typename From>
struct element_caster<To, From> {
  [[nodiscard]] static constexpr auto operator()(From value) -> To {
    using representation = typename std::remove_cvref_t<To>::rep;

    static_assert(std::same_as<representation, std::remove_cvref_t<From>>,
                  "The underlying storage type must be identical to the "
                  "quantity representation type to guarantee the conversion is "
                  "explicitly decided by the end-user.");

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
                  "explicitly decided by the end-user.");

    return value * To::reference;
  }
};

template <typename To, mp_units::QuantityPoint From>
struct element_caster<To, From> {
  [[nodiscard]] static constexpr auto operator()(From value) -> To {
    using representation = typename std::remove_cvref_t<From>::rep;

    static_assert(std::same_as<representation, std::remove_cvref_t<To>>,
                  "The underlying storage type must be identical to the "
                  "quantity representation type to guarantee the conversion is "
                  "explicitly decided by the end-user.");

    return value.quantity_from_zero().numerical_value_in(value.unit);
  }
};

template <mp_units::QuantityPoint To, typename From>
struct element_caster<To, From> {
  [[nodiscard]] static constexpr auto operator()(From value) -> To {
    using representation = typename std::remove_cvref_t<To>::rep;

    static_assert(std::same_as<representation, std::remove_cvref_t<From>>,
                  "The underlying storage type must be identical to the "
                  "quantity representation type to guarantee the conversion is "
                  "explicitly decided by the end-user.");

    return {value * To::unit, mp_units::default_point_origin(To::unit)};
  }
};

template <mp_units::QuantityPoint To, typename From>
struct element_caster<To &, From &> {
  // A quantity point reference cannot be safely materialized out of a
  // representation reference. It would be undefined behavior even if the size,
  // padding, alignment, aliasing are controlled. Therefore the best we can do
  // is to return a constant quantity value to inform the end-user lvalue
  // reference assignment cannot be supported.
  [[nodiscard]] static constexpr auto operator()(From value) -> const To {
    using representation = typename std::remove_cvref_t<To>::rep;

    static_assert(std::same_as<representation, std::remove_cvref_t<From>>,
                  "The underlying storage type must be identical to the "
                  "quantity representation type to guarantee the conversion is "
                  "explicitly decided by the end-user.");

    return {value * To::unit, mp_units::default_point_origin(To::unit)};
  }
};

template <typename To, mp_units::Reference From>
struct element_caster<To, From> {
  [[nodiscard]] static constexpr auto operator()([[maybe_unused]] From value)
      -> To {
    return 1.;
  }
};

template <typename Type, mp_units::UnitMagnitude Magnitude>
struct multiplies<Type, Magnitude> {
  // A type that provides its own magnitude-aware operator*(T, UnitMagnitude)
  // customization point. scale() will prefer this path when available, and the
  // return type may differ from the input (e.g. a wrapper with scaled bounds).
  [[nodiscard]] static constexpr auto operator()(const Type &lhs,
                                                 const Magnitude &rhs) -> Type;
};

// True when at least one of the matrix's logical elements (`row index x
// column index`, per element) is itself an mp-units quantity, as opposed to a
// plain arithmetic representation (e.g. `double`). A homogeneous,
// plain-representation typed vector or matrix (e.g. `column_vector<double,
// double, double, double>`) is a legitimate mp-units Representation.
// A typed matrix whose elements are
// themselves quantities (e.g. a per-axis heterogeneous velocity vector, or
// any rank-0 "singleton" wrapping a single quantity) is a *result* of
// decomposing a quantity, not a candidate storage for one; see
// `disable_representation` below for why it must say so. The element types are
// deduced through the public typed linear algebra interface only: the product
// of the row index and the column index at each position.
template <typename Type, std::size_t Row, std::size_t... Columns>
[[nodiscard]] consteval auto has_quantity_element_in_row(
    [[maybe_unused]] std::index_sequence<Columns...> columns) -> bool {
  using matrix = std::remove_cvref_t<Type>;
  using row_index = std::tuple_element_t<Row, typename matrix::row_indexes>;

  return (
      mp_units::Quantity<std::remove_cvref_t<std::invoke_result_t<
          multiplies<row_index, std::tuple_element_t<
                                    Columns, typename matrix::column_indexes>>,
          row_index,
          std::tuple_element_t<Columns, typename matrix::column_indexes>>>> or
      ...);
}

template <typename Type, std::size_t... Rows>
[[nodiscard]] consteval auto
has_quantity_element([[maybe_unused]] std::index_sequence<Rows...> rows)
    -> bool {
  return (has_quantity_element_in_row<Type, Rows>(
              std::make_index_sequence<std::remove_cvref_t<Type>::columns>{}) or
          ...);
}

template <typename Type>
concept quantity_element_typed_matrix =
    same_as_typed_matrix<Type> and
    has_quantity_element<Type>(
        std::make_index_sequence<std::remove_cvref_t<Type>::rows>{});
} // namespace fcarouge

namespace mp_units {
// A typed_matrix is a composite facade over a backend matrix, never itself a
// quantity's scalar/vector/tensor storage (its own elements, individually,
// fill that role): opt it out of representation status exactly when its
// elements are themselves quantities. Left unguarded, mp-units' generic
// representation detection (RepresentationOf, consulted by quantity's own
// operator==, operator+, and friends when overload resolution considers a
// typed_matrix-of-quantities operand) would classify that matrix's character
// by evaluating its operators; those operators are in turn defined in terms
// of the matrix's own quantity elements, which recurses back into the same
// RepresentationOf query already in flight for the matrix itself. That
// self-reference is what surfaces as "satisfaction of constraint
// 'RealScalar<T>' depends on itself". Declining representation status
// upfront, the same guard mp-units applies to its own `quantity` (see
// `disable_representation`'s default and `RepresentationBaseline`), stops
// the character check, and therefore the cycle, before it starts.
//
// A typed_matrix of plain, non-quantity elements (e.g. a homogeneous
// `column_vector<double, double, double, double>`) is left enabled: it is
// exactly the kind of representation `RepresentationOf` is meant to detect,
// and its element-wise operators only ever involve plain arithmetic, so no
// cycle is possible.
template <typename Matrix, typename RowIndexes, typename ColumnIndexes>
inline constexpr bool disable_representation<
    fcarouge::typed_matrix<Matrix, RowIndexes, ColumnIndexes>> =
    fcarouge::quantity_element_typed_matrix<
        fcarouge::typed_matrix<Matrix, RowIndexes, ColumnIndexes>>;
} // namespace mp_units

#endif // FCAROUGE_MP_UNITS_HPP
