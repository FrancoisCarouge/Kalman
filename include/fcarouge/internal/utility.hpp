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

#ifndef FCAROUGE_INTERNAL_UTILITY_HPP
#define FCAROUGE_INTERNAL_UTILITY_HPP

#include <concepts>
#include <tuple>
#include <type_traits>

namespace fcarouge::internal {
template <typename Type>
concept kalman_filter = requires(Type value) {
  //! @todo What should be a better concept of the Kalman filter of this
  //! library?
  typename Type::state;
  typename Type::output;
};

template <typename Type>
concept arithmetic = std::integral<Type> || std::floating_point<Type>;

template <typename Type>
concept algebraic = not arithmetic<Type>;

template <typename Type>
concept eigen = requires { typename Type::PlainMatrix; };

template <typename Filter>
concept has_input_member = requires(Filter filter) { filter.u; };

//! @todo Is the _method concept extraneous or incorrect? Explain the
//! shortcoming?

template <typename Filter>
concept has_input_method = requires(Filter filter) { filter.u(); };

//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_input = has_input_member<Filter> || has_input_method<Filter>;

template <typename Filter>
concept has_process_uncertainty_member = requires(Filter filter) { filter.q; };

template <typename Filter>
concept has_process_uncertainty_method =
    requires(Filter filter) { filter.q(); };

//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_process_uncertainty = has_process_uncertainty_member<Filter> ||
                                  has_process_uncertainty_method<Filter>;

template <typename Filter>
concept has_output_uncertainty_member = requires(Filter filter) { filter.r; };

template <typename Filter>
concept has_output_uncertainty_method = requires(Filter filter) { filter.r(); };

//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_output_uncertainty = has_output_uncertainty_member<Filter> ||
                                 has_output_uncertainty_method<Filter>;

template <typename Filter>
concept has_input_control_member = requires(Filter filter) { filter.g; };

template <typename Filter>
concept has_input_control_method = requires(Filter filter) { filter.g(); };

//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_input_control =
    has_input_control_member<Filter> || has_input_control_method<Filter>;

template <typename Filter>
concept has_state_transition_member = requires(Filter filter) { filter.f; };

template <typename Filter>
concept has_state_transition_method = requires(Filter filter) { filter.f(); };

//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_state_transition =
    has_state_transition_member<Filter> || has_state_transition_method<Filter>;

template <typename Filter>
concept has_output_model_member = requires(Filter filter) { filter.h; };

template <typename Filter>
concept has_output_model_method = requires(Filter filter) { filter.h(); };

//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_output_model =
    has_output_model_member<Filter> || has_output_model_method<Filter>;

template <typename Filter>
concept has_prediction_types =
    requires() { typename Filter::prediction_types; };

template <typename Filter>
concept has_update_types = requires() { typename Filter::update_types; };

// There is only one known way to do conditional member types: partial
// specialization of class templates.
template <typename Filter> struct conditional_input {};

template <has_input Filter> struct conditional_input<Filter> {
  //! @brief Type of the control column vector U.
  //!
  //! @details This member type is conditionally present. The presence of the
  //! member depends on the filter capabilities.
  using input = Filter::input;
};

template <typename Filter> struct conditional_input_control {};

template <has_input_control Filter> struct conditional_input_control<Filter> {
  //! @brief Type of the control transition matrix G.
  //!
  //! @details Also known as B. This member type is conditionally present. The
  //! presence of the member depends on the filter capabilities.
  using input_control = Filter::input_control;
};

template <typename Filter> struct conditional_output_model {};

template <has_output_model Filter> struct conditional_output_model<Filter> {
  //! @brief Type of the observation transition matrix H.
  //!
  //! @details Also known as the measurement transition matrix or C. The
  //! presence of the member depends on the filter capabilities.
  using output_model = Filter::output_model;
};

template <typename Filter> struct conditional_process_uncertainty {};

template <has_process_uncertainty Filter>
struct conditional_process_uncertainty<Filter> {
  //! @brief Type of the process noise correlated variance matrix Q.
  using process_uncertainty = Filter::process_uncertainty;
};

template <typename Filter> struct conditional_output_uncertainty {};

template <has_output_uncertainty Filter>
struct conditional_output_uncertainty<Filter> {
  //! @brief Type of the observation noise correlated variance matrix R.
  using output_uncertainty = Filter::output_uncertainty;
};

template <typename Filter> struct conditional_prediction_types {};

template <has_prediction_types Filter>
struct conditional_prediction_types<Filter> {
  //! @brief Pack of the prediction parameters.
  using prediction_types = Filter::prediction_types;
};

template <typename Filter> struct conditional_state_transition {};

template <has_state_transition Filter>
struct conditional_state_transition<Filter> {
  //! @brief Type of the state transition matrix F.
  //!
  //! @details Also known as the fundamental matrix, propagation, Φ, or A.
  using state_transition = Filter::state_transition;
};

template <typename Filter> struct conditional_update_types {};

template <has_update_types Filter> struct conditional_update_types<Filter> {
  //! @brief Pack of the update parameters.
  using update_types = Filter::update_types;
};

// The only way to have a conditional member type is to inherit from a template
// specialization on the member type.
template <typename Filter>
struct conditional_member_types : public conditional_input_control<Filter>,
                                  conditional_input<Filter>,
                                  conditional_output_model<Filter>,
                                  conditional_output_uncertainty<Filter>,
                                  conditional_prediction_types<Filter>,
                                  conditional_process_uncertainty<Filter>,
                                  conditional_state_transition<Filter>,
                                  conditional_update_types<Filter> {};

template <typename Type> struct repack {
  using type = Type;
};

template <template <typename...> typename Pack, typename... Types>
struct repack<Pack<Types...>> {
  using type = std::tuple<Types...>;
  using size_t = std::remove_const_t<decltype(sizeof...(Types))>;

  static inline constexpr auto size{sizeof...(Types)};
};

template <auto Value, auto... Values> struct first_value {
  static constexpr auto value{Value};
};

template <typename Type> struct not_implemented {
  template <auto Size>
  inline constexpr explicit not_implemented(
      [[maybe_unused]] const char (&message)[Size]) {
    // The argument message is printed in the compiler error output.
  }

  static constexpr auto type_dependent_false{sizeof(Type) != sizeof(Type)};
  static constexpr auto missing{type_dependent_false};

  static_assert(missing, "This type is not implemented. See compiler message.");
};

struct transposer final {
  template <arithmetic Arithmetic>
  [[nodiscard]] inline constexpr auto
  operator()(const Arithmetic &value) const {
    return value;
  }

  template <typename Matrix>
    requires requires(Matrix value) { value.transpose(); }
  [[nodiscard]] inline constexpr auto operator()(const Matrix &value) const {
    return value.transpose();
  }

  template <typename Matrix>
    requires requires(Matrix value) { transpose(value); }
  [[nodiscard]] inline constexpr auto operator()(const Matrix &value) const {
    return transpose(value);
  }
};

struct matrix_deducer;

template <typename Lhs, typename Rhs>
using deduce_matrix =
    std::remove_cvref_t<std::invoke_result_t<matrix_deducer, Lhs, Rhs>>;

struct matrix_deducer final {
  template <typename Lhs, typename Rhs>
  [[nodiscard]] inline constexpr auto
  operator()(Lhs lhs, Rhs rhs) const -> decltype(lhs * transposer{}(rhs));

  template <typename Lhs, typename Rhs>
    requires(eigen<Lhs> or eigen<Rhs>)
  [[nodiscard]] inline constexpr auto operator()(Lhs lhs, Rhs rhs) const ->
      typename decltype(lhs * transposer{}(rhs))::PlainMatrix;
};

using empty_tuple = std::tuple<>;

template <typename Pack> using repack_t = repack<Pack>::type;

template <typename Pack> using size_t = repack<Pack>::size_t;

template <typename... Types>
using first_t = std::tuple_element_t<0, std::tuple<Types...>>;

template <auto Begin, decltype(Begin) End, decltype(Begin) Increment,
          typename Function>
constexpr void for_constexpr(Function &&function) {
  if constexpr (Begin < End) {
    function(std::integral_constant<decltype(Begin), Begin>());
    for_constexpr<Begin + Increment, End, Increment>(function);
  }
}

template <typename Pack> inline constexpr auto size{repack<Pack>::size};

template <auto... Values>
inline constexpr auto first_v{first_value<Values...>::value};
} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_UTILITY_HPP
