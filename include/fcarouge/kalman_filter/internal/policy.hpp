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

#ifndef FCAROUGE_KALMAN_FILTER_INTERNAL_POLICY_HPP
#define FCAROUGE_KALMAN_FILTER_INTERNAL_POLICY_HPP

#include "type.hpp"
#include "utility.hpp"

#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace fcarouge::kalman_filter::internal {
template <typename Type>
concept update_form =
    std::same_as<Type, joseph_form_t> || std::same_as<Type, simple_form_t> ||
    std::same_as<Type, symmetric_form_t> ||
    std::same_as<Type, information_form_t>;

template <typename Type> struct is_floors_policy : std::false_type {};

template <typename... Types>
struct is_floors_policy<floors<Types...>> : std::true_type {};

template <typename Type>
concept floors_policy = is_floors_policy<Type>::value;

// A compile-time constant: an empty type with a static value member, as the
// standard constant wrapper and integral constant.
template <typename Type>
concept constant = std::is_empty_v<Type> && requires { Type::value; };

template <typename Type> struct stores_floors : std::false_type {};

template <typename... Types>
struct stores_floors<floors<Types...>>
    : std::bool_constant<(not constant<Types> || ...)> {};

// The floors with a runtime standard deviation, stored in the filter.
template <typename Type>
concept stored_floors = stores_floors<Type>::value;

// The type of the standard deviation, of a value or of a constant.
template <typename Type> struct deviation : std::type_identity<Type> {};

template <constant Type>
struct deviation<Type>
    : std::type_identity<std::remove_cvref_t<decltype(Type::value)>> {};

template <typename Type> using deviation_t = deviation<Type>::type;

// The standard deviation of a value or of a constant.
template <typename Type>
[[nodiscard]] constexpr auto deviation_of(const Type &value)
    -> deviation_t<Type> {
  if constexpr (constant<Type>) {
    return Type::value;
  } else {
    return value;
  }
}

template <typename Type>
concept policy = update_form<Type> || std::same_as<Type, symmetrize_t> ||
                 floors_policy<Type>;

// The policies of a filter, in their canonical order: the update form, the
// symmetrization, then the floors. The filter holds its policies as its first
// member, without a unique address, which stores nothing but the floors with a
// runtime standard deviation: a policy declared or not is part of the filter
// type, at no space cost, and its absence selects the default filter.
template <typename... Policies> struct policies {};

template <typename... Policies>
  requires(stored_floors<Policies> || ...)
struct policies<Policies...> {
  decltype(std::tuple_element_t<sizeof...(Policies) - 1,
                                std::tuple<Policies...>>::value) deviations;
};

template <typename Type, template <typename> typename Predicate>
using select =
    std::conditional_t<Predicate<Type>::value, std::tuple<Type>, std::tuple<>>;

template <typename Type>
using is_update_form = std::bool_constant<update_form<Type>>;

// The update forms other than the default Joseph form: declaring the default
// form explicitly is the default filter, of the same type.
template <typename Type>
using is_other_update_form =
    std::bool_constant<update_form<Type> &&
                       not std::same_as<Type, joseph_form_t>>;

template <typename Type> using is_symmetrize = std::is_same<Type, symmetrize_t>;

template <typename Type>
using is_floors = std::bool_constant<floors_policy<Type>>;

template <typename Type> using is_policy = std::bool_constant<policy<Type>>;

template <typename Type>
using is_characteristic = std::bool_constant<not policy<Type>>;

template <typename Tuple> struct policies_from;

template <typename... Policies>
struct policies_from<std::tuple<Policies...>>
    : std::type_identity<policies<Policies...>> {};

// The canonical policies of the declaration arguments: the order of declaration
// of the policies does not matter.
template <typename... Arguments>
using policies_of = policies_from<decltype(std::tuple_cat(
    std::declval<select<Arguments, is_other_update_form>>()...,
    std::declval<select<Arguments, is_symmetrize>>()...,
    std::declval<select<Arguments, is_floors>>()...))>::type;

// The declaration arguments other than the policies.
template <typename... Arguments>
using characteristics_of = decltype(std::tuple_cat(
    std::declval<select<Arguments, is_characteristic>>()...));

template <template <typename> typename Predicate, typename... Arguments>
inline constexpr std::size_t count_of{
    (0U + ... + std::size_t{Predicate<Arguments>::value})};

template <typename Policy, typename Policies>
inline constexpr bool contains{false};

template <typename Policy, typename... Policies>
inline constexpr bool contains<Policy, policies<Policies...>>{
    (std::same_as<Policy, Policies> || ...)};

template <std::size_t Index>
inline constexpr std::integral_constant<std::size_t, Index> position{};

// A matrix whose elements are accessed at compile-time positions, by the
// multidimensional subscript or the function call operators: no backend
// customization point, and heterogeneous typed matrices are supported.
template <typename Type>
concept indexable = requires(Type value) { value[position<0>, position<0>]; } ||
                    requires(Type value) { value(position<0>, position<0>); };

template <std::size_t Row, std::size_t Column>
[[nodiscard]] constexpr decltype(auto) at(auto &value) {
  if constexpr (requires { value[position<Row>, position<Column>]; }) {
    return value[position<Row>, position<Column>];
  } else {
    return value(position<Row>, position<Column>);
  }
}

// Averages the estimate uncertainty with its transpose. A scalar is symmetric.
template <typename Type> constexpr void symmetrize_estimate(Type &p) {
  if constexpr (indexable<Type>) {
    using element = std::remove_cvref_t<decltype(at<0, 0>(p))>;

    if constexpr (std::floating_point<element>) {
      p = Type{(p + t(p)) * element{0.5}};
    } else {
      p = Type{(p + t(p)) * 0.5};
    }
  }
}

// Raises the variance to the square of its standard deviation floor.
constexpr void raise_variance(auto &variance, const auto &deviation) {
  const auto floor{deviation * deviation};

  if (variance < floor) {
    variance = floor;
  }
}

// Raises each diagonal variance of the estimate uncertainty, in state order.
template <typename Type, typename... Deviations>
constexpr void raise_variances(Type &p,
                               const std::tuple<Deviations...> &deviations) {
  if constexpr (indexable<Type>) {
    static_assert(std::is_lvalue_reference_v<decltype(at<0, 0>(p))>,
                  "The floors require assignable estimate uncertainty "
                  "elements from the linear algebra backend.");

    [&]<std::size_t... Indexes>(std::index_sequence<Indexes...>) {
      (raise_variance(at<Indexes, Indexes>(p),
                      deviation_of(std::get<Indexes>(deviations))),
       ...);
    }(std::index_sequence_for<Deviations...>{});
  } else {
    raise_variance(p, deviation_of(std::get<0>(deviations)));
  }
}

// Conditions the estimate uncertainty after an update or a prediction.
template <typename... Policies>
constexpr void condition(const policies<Policies...> &configuration, auto &p) {
  if constexpr ((std::same_as<Policies, symmetrize_t> || ...)) {
    symmetrize_estimate(p);
  }

  if constexpr (sizeof...(Policies) > 0) {
    using last =
        std::tuple_element_t<sizeof...(Policies) - 1, std::tuple<Policies...>>;

    if constexpr (stored_floors<last>) {
      raise_variances(p, configuration.deviations);
    } else if constexpr (floors_policy<last>) {
      raise_variances(p, decltype(last::value){});
    }
  }
}

// Updates the estimate uncertainty of a filter with an output model H, then
// conditions it. The forms agree in exact arithmetic, the gain being optimal.
// The simple form of a scalar output and a matrix state is associated as
// P - K(HP), and the default Joseph form as (P - K(HP)) - (P - K(HP))HᵀKᵀ +
// KRKᵀ: the same products, at a quadratic rather than a cubic cost in the state
// dimension.
template <typename... Policies>
constexpr void
update_estimate_uncertainty(const policies<Policies...> &configuration, auto &p,
                            const auto &i, const auto &k, const auto &h,
                            const auto &r, const auto &s) {
  using estimate_uncertainty = std::remove_cvref_t<decltype(p)>;
  using kinds = policies<Policies...>;

  constexpr bool scalar_output{indexable<estimate_uncertainty> &&
                               not indexable<std::remove_cvref_t<decltype(s)>>};

  if constexpr (contains<simple_form_t, kinds> && scalar_output) {
    p = estimate_uncertainty{p - k * (h * p)};
  } else if constexpr (contains<simple_form_t, kinds>) {
    p = estimate_uncertainty{(i - k * h) * p};
  } else if constexpr (contains<symmetric_form_t, kinds>) {
    p = estimate_uncertainty{p - k * s * t(k)};
  } else if constexpr (contains<information_form_t, kinds>) {
    p = estimate_uncertainty{i / (i / p + t(h) / r * h)};
  } else if constexpr (scalar_output) {
    p = estimate_uncertainty{p - k * (h * p)};
    p = estimate_uncertainty{p - p * t(h) * t(k) + k * r * t(k)};
  } else {
    p = estimate_uncertainty{(i - k * h) * p * t(i - k * h) + k * r * t(k)};
  }

  condition(configuration, p);
}

// Updates the estimate uncertainty of a filter without an output model, the
// identity, then conditions it.
template <typename... Policies>
constexpr void
update_estimate_uncertainty(const policies<Policies...> &configuration, auto &p,
                            const auto &i, const auto &k, const auto &r,
                            const auto &s) {
  using estimate_uncertainty = std::remove_cvref_t<decltype(p)>;
  using kinds = policies<Policies...>;

  if constexpr (contains<simple_form_t, kinds>) {
    p = estimate_uncertainty{(i - k) * p};
  } else if constexpr (contains<symmetric_form_t, kinds>) {
    p = estimate_uncertainty{p - k * s * t(k)};
  } else if constexpr (contains<information_form_t, kinds>) {
    p = estimate_uncertainty{i / (i / p + i / r)};
  } else {
    p = estimate_uncertainty{(i - k) * p * t(i - k) + k * r * t(k)};
  }

  condition(configuration, p);
}
} // namespace fcarouge::kalman_filter::internal

#endif // FCAROUGE_KALMAN_FILTER_INTERNAL_POLICY_HPP
