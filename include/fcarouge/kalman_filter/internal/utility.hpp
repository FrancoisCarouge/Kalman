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

#ifndef FCAROUGE_KALMAN_FILTER_INTERNAL_UTILITY_HPP
#define FCAROUGE_KALMAN_FILTER_INTERNAL_UTILITY_HPP

#include <concepts>
#include <cstddef>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

namespace fcarouge::kalman_filter::internal {
//! @name Concepts
//! @{

//! @brief Arithmetic concept.
//!
//! @details Any integer or floating point type.
template <typename Type>
concept arithmetic = std::integral<Type> || std::floating_point<Type>;

//! @brief Algebraic concept.
//!
//! @details Not an arithmetic type.
//!
//! @todo What should be a better concept of an algebraic type?
template <typename Type>
concept algebraic = requires(Type value) { value(0, 0); };

//! @brief The value copy-initializes the type without narrowing.
//!
//! @details The initialization of an array element from a braced list, where
//! narrowing conversions and explicit constructors do not participate.
template <typename Value, typename Type>
concept converts_without_narrowing =
    requires(Value value) { std::type_identity_t<Type[1]>{value}; };

//! @brief Deduces the type of the element at the position of the type.
//!
//! @details A backend names the element type of its matrices its own way, if
//! at all: the type is deduced from the element access instead. The
//! compile-time indexed access `at<Position>()` of a typed matrix, whose
//! elements may each have their own type, otherwise the call operator
//! `(Position)` of a matrix whose elements share a type. Void for a type
//! without indexed element access, an arithmetic type for one.
template <typename Type, std::size_t Position>
[[nodiscard]] consteval auto deduce_element() {
  if constexpr (requires(const Type value) { value.template at<Position>(); }) {
    return std::type_identity<std::remove_cvref_t<
        decltype(std::declval<const Type &>().template at<Position>())>>{};
  } else if constexpr (requires(const Type value) { value(Position); }) {
    return std::type_identity<std::remove_cvref_t<
        decltype(std::declval<const Type &>()(Position))>>{};
  } else {
    return std::type_identity<void>{};
  }
}

//! @brief The type of the element at the position of the type.
template <typename Type, std::size_t Position>
using element = decltype(deduce_element<Type, Position>())::type;

//! @brief Whether each value initializes the element at its position of the
//! type without narrowing, where the type of the element is deduced.
template <typename Type, typename... Values, std::size_t... Positions>
[[nodiscard]] consteval auto
converts_elements(std::index_sequence<Positions...> /*positions*/) -> bool {
  return ((std::same_as<element<Type, Positions>, void> ||
           converts_without_narrowing<Values, element<Type, Positions>>) &&
          ...);
}

//! @brief The values initialize the elements of the type without narrowing.
//!
//! @details Several values initialize the type from its elements, one value
//! per position. The constructor of a backend matrix converts the elements
//! itself, narrowing included, and its constructor template accepts any
//! element type: each value is checked against the element at its position
//! instead, as a built-in arithmetic characteristic is. A single value
//! initializes the type as a whole and is not an element.
template <typename Type, typename... Values>
concept initializes_elements =
    sizeof...(Values) < 2 ||
    converts_elements<Type, Values...>(std::index_sequence_for<Values...>{});

//! @brief The values initialize a characteristic of the type, as opposed to
//! its callable.
//!
//! @details A single value initializes the characteristic by implicit
//! conversion, and not by construction: a backend matrix may be explicitly
//! constructible from any type, a callable included. Several values initialize
//! the characteristic from its elements, by list-initialization. Either way,
//! the values direct-list-initialize the characteristic, without narrowing,
//! of the characteristic or of its elements.
template <typename Type, typename... Values>
concept initializes =
    (sizeof...(Values) > 1 || (sizeof...(Values) == 1 &&
                               (std::is_convertible_v<Values, Type> && ...))) &&
    requires(Values... values) { Type{values...}; } &&
    initializes_elements<Type, Values...>;

//! @brief The values direct-list-initialize the type.
//!
//! @details Explicit constructors participate, narrowing conversions do not,
//! of the type or of its elements: the expression the filter evaluates to
//! store the values.
template <typename Type, typename... Values>
concept list_initializes = sizeof...(Values) > 0 && requires(Values... values) {
  Type{values...};
} && initializes_elements<Type, Values...>;

//! @brief The object, a reference to it, is not constant.
template <typename Object>
concept modifiable = not std::is_const_v<std::remove_reference_t<Object>>;

//! @brief The values, if any, set a characteristic of the type through the
//! object.
//!
//! @details A read takes no values. A write needs a modifiable object.
template <typename Object, typename Type, typename... Values>
concept sets = sizeof...(Values) == 0 ||
               (modifiable<Object> && list_initializes<Type, Values...>);

//! @brief The values set a characteristic of the type, as opposed to its
//! callable, through the modifiable object.
template <typename Object, typename Type, typename... Values>
concept sets_value = modifiable<Object> && initializes<Type, Values...>;

//! @brief The values replace a callable characteristic through the modifiable
//! object.
//!
//! @details Name the callable type in a separate alternative of the
//! constraint: for a characteristic without a callable, the substitution fails
//! for this concept alone, and not for the alternatives it is combined with.
template <typename Object, typename Callable, typename... Values>
concept replaces =
    sizeof...(Values) > 0 &&
    modifiable<Object> &&requires(Values... values) { Callable{values...}; };

//! @brief The filter holds the prediction argument at the position.
template <typename Filter, std::size_t Position>
concept has_prediction_argument = requires(Filter filter) {
  requires Position < std::tuple_size_v<decltype(filter.prediction_arguments)>;
};

//! @brief The filter holds the update argument at the position.
template <typename Filter, std::size_t Position>
concept has_update_argument = requires(Filter filter) {
  requires Position < std::tuple_size_v<decltype(filter.update_arguments)>;
};

template <typename Filter>
concept has_state_member = requires(Filter filter) { filter.x; };

//! @todo Is the _method concept extraneous or incorrect? Explain the
//! shortcoming?
template <typename Filter>
concept has_state_method = requires(Filter filter) { filter.x(); };

//! @brief Filter state support concept.
//!
//! @details The filter supports the state related functionality: `state` type
//! member and `x()` method.
//!
//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_state = has_state_member<Filter> || has_state_method<Filter>;

template <typename Filter>
concept has_output_member = requires(Filter filter) { filter.z; };

//! @todo Is the _method concept extraneous or incorrect? Explain the
//! shortcoming?
template <typename Filter>
concept has_output_method = requires(Filter filter) { filter.z(); };

//! @brief Filter output support concept.
//!
//! @details The filter supports the output related functionality: `output` type
//! member and `z()` method.
//!
//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_output = has_output_member<Filter> || has_output_method<Filter>;

template <typename Filter>
concept has_input_member = requires(Filter filter) { filter.u; };

//! @todo Is the _method concept extraneous or incorrect? Explain the
//! shortcoming?
template <typename Filter>
concept has_input_method = requires(Filter filter) { filter.u(); };

//! @brief Filter input support concept.
//!
//! @details The filter supports the input related functionality: `input` type
//! member and `u()` method.
//!
//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_input = has_input_member<Filter> || has_input_method<Filter>;

template <typename Filter>
concept has_estimate_uncertainty_member = requires(Filter filter) { filter.p; };

template <typename Filter>
concept has_estimate_uncertainty_method =
    requires(Filter filter) { filter.p(); };

//! @brief Filter estimate uncertainty support concept.
//!
//! @details The filter supports the estimate uncertainty related functionality:
//! `estimate_uncertainty` type member and `p()` method.
//!
//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_estimate_uncertainty = has_estimate_uncertainty_member<Filter> ||
                                   has_estimate_uncertainty_method<Filter>;

template <typename Filter>
concept has_process_uncertainty_member = requires(Filter filter) { filter.q; };

template <typename Filter>
concept has_process_uncertainty_method =
    requires(Filter filter) { filter.q(); };

//! @brief Filter process uncertainty support concept.
//!
//! @details The filter supports the process uncertainty related functionality:
//! `process_uncertainty` type member and `q()` method.
//!
//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_process_uncertainty = has_process_uncertainty_member<Filter> ||
                                  has_process_uncertainty_method<Filter>;

template <typename Filter>
concept has_output_uncertainty_member = requires(Filter filter) { filter.r; };

template <typename Filter>
concept has_output_uncertainty_method = requires(Filter filter) { filter.r(); };

//! @brief Filter output uncertainty support concept.
//!
//! @details The filter supports the output uncertainty related functionality:
//! `output_uncertainty` type member and `r()` method.
//!
//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_output_uncertainty = has_output_uncertainty_member<Filter> ||
                                 has_output_uncertainty_method<Filter>;

//! @brief Filter prediction pack support concept.
//!
//! @details The filter supports the prediction parameters related
//! functionality: `prediction_types` type member and parameters for the
//! `predict()` method.
template <typename Filter>
concept has_prediction_types =
    requires() { typename Filter::prediction_types; };

template <typename Filter>
concept has_input_control_member = requires(Filter filter) { filter.g; };

template <typename Filter>
concept has_input_control_method = requires(Filter filter) { filter.g(); };

//! @brief Filter input control support concept.
//!
//! @details The filter supports the input control related functionality:
//! `input_control` type member and `g()` method.
//!
//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_input_control =
    has_input_control_member<Filter> || has_input_control_method<Filter>;

template <typename Filter>
concept has_state_transition_member = requires(Filter filter) { filter.f; };

template <typename Filter>
concept has_state_transition_method = requires(Filter filter) { filter.f(); };

//! @brief Filter state transition support concept.
//!
//! @details The filter supports the state transition related functionality:
//! `state_transition` type member and `f()` method.
//!
//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_state_transition =
    has_state_transition_member<Filter> || has_state_transition_method<Filter>;

//! @brief Filter update pack support concept.
//!
//! @details The filter supports the update parameters related functionality:
//! `update_types` type member and parameters for the `update()` method.
template <typename Filter>
concept has_update_types = requires() { typename Filter::update_types; };

template <typename Filter>
concept has_output_model_member = requires(Filter filter) { filter.h; };

template <typename Filter>
concept has_output_model_method = requires(Filter filter) { filter.h(); };

//! @brief Filter output model support concept.
//!
//! @details The filter supports the output model related functionality:
//! `output_model` type member and `h()` method.
//!
//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_output_model =
    has_output_model_member<Filter> || has_output_model_method<Filter>;

template <typename Filter>
concept has_gain_member = requires(Filter filter) { filter.k; };

template <typename Filter>
concept has_gain_method = requires(Filter filter) { filter.k(); };

//! @brief Filter gain support concept.
//!
//! @details The filter supports the output model related functionality:
//! `gain` type member and `k()` method.
//!
//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_gain = has_gain_member<Filter> || has_gain_method<Filter>;

template <typename Filter>
concept has_innovation_member = requires(Filter filter) { filter.y; };

template <typename Filter>
concept has_innovation_method = requires(Filter filter) { filter.y(); };

//! @brief Filter innovation support concept.
//!
//! @details The filter supports the innovation related functionality:
//! `innovation` type member and `y()` method.
//!
//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_innovation =
    has_innovation_member<Filter> || has_innovation_method<Filter>;

template <typename Filter>
concept has_innovation_uncertainty_member =
    requires(Filter filter) { filter.s; };

template <typename Filter>
concept has_innovation_uncertainty_method =
    requires(Filter filter) { filter.s(); };

//! @brief Filter innovation uncertainty support concept.
//!
//! @details The filter supports the innovation uncertainty related
//! functionality: `innovation_uncertainty` type member and `s()` method.
//!
//! @todo Shorten when MSVC has better if-constexpr-requires support.
template <typename Filter>
concept has_innovation_uncertainty =
    has_innovation_uncertainty_member<Filter> ||
    has_innovation_uncertainty_method<Filter>;

//! @}

//! @name Types
//! @{

//! @brief Linear algebra divides expression type specialization point.
//!
//! @details Matrix division is a mathematical abuse of terminology. Informally
//! defined as multiplication by the inverse. Similarly to division by zero in
//! real numbers, there exists matrices that are not invertible. Remember the
//! division operation is not commutative. Matrix inversion can be avoided by
//! solving `X * rhs = lhs` for `rhs` through a decomposer. There exists several
//! ways to decompose and solve the equation. Implementations trade off
//! numerical stability, triangularity, symmetry, space, time, etc. Dividing an
//! `R1 x C` matrix by an `R2 x C` matrix results in an `R1 x R2` matrix.
template <typename Lhs, typename Rhs> struct divides {
  [[nodiscard]] static constexpr auto operator()(const Lhs &lhs, const Rhs &rhs)
      -> decltype(lhs / rhs);
};

//! @brief Divider helper type.
template <typename Lhs, typename Rhs>
using quotient =
    std::invoke_result_t<divides<Lhs, Rhs>, const Lhs &, const Rhs &>;

//! @brief Linear algebra evaluates override expression lazy evaluation
//! specialization point.
template <typename Type> struct evaluates {
  [[nodiscard]] static constexpr auto operator()() -> Type;
};

//! @brief Evaluator helper type.
template <typename Type> using evaluate = std::invoke_result_t<evaluates<Type>>;

//! @brief Linear algebra transposes specialization point.
template <typename Type> struct transposes {
  [[nodiscard]] static constexpr auto operator()(const Type &value) {
    return value;
  }
};

template <typename Type>
  requires requires(Type value) { value.transpose(); }
struct transposes<Type> {
  [[nodiscard]] static constexpr auto operator()(const Type &value) {
    return value.transpose();
  }
};

//! @brief Transposer helper type.
template <typename Type>
using transpose = std::invoke_result_t<transposes<Type>, const Type &>;

//! @brief Transpose helper function.
//!
//! @details Enable readable linear algebra notation.
template <typename Type> constexpr auto t(const Type &value) {
  return transposes<Type>{}(value);
}

//! @brief Unpack the first type of the type template parameter pack.
//!
//! @details Shorthand for `std::tuple_element_t<0, std::tuple<Types...>>`.
template <typename... Types>
using first = std::tuple_element_t<0, std::tuple<Types...>>;

template <typename Type, std::size_t Size> struct tupler {
  template <typename = std::make_index_sequence<Size>> struct helper;

  template <std::size_t... Indexes>
  struct helper<std::index_sequence<Indexes...>> {
    template <std::size_t> using wrap = Type;

    using type = std::tuple<wrap<Indexes>...>;
  };

  using type = typename helper<>::type;
};

//! @brief An alias for making a tuple of the same type.
template <typename Type, std::size_t Size>
using tuple_n_type = typename tupler<Type, Size>::type;

//! @brief Type multiplies expression type specialization point.
template <typename Lhs, typename Rhs> struct multiplies {
  [[nodiscard]] static constexpr auto operator()(const Lhs &lhs, const Rhs &rhs)
      -> decltype(lhs * rhs);
};

//! @brief Helper type to deduce the result type of the product.
template <typename Lhs, typename Rhs>
using product =
    std::invoke_result_t<multiplies<Lhs, Rhs>, const Lhs &, const Rhs &>;

//! @brief Type minus, subtraction expression type specialization point.
template <typename Lhs, typename Rhs> struct minus {
  [[nodiscard]] static constexpr auto operator()(const Lhs &lhs, const Rhs &rhs)
      -> decltype(lhs - rhs);
};

//! @brief Helper type to deduce the result type of the minus, subtraction.
template <typename Lhs, typename Rhs>
using difference =
    std::invoke_result_t<minus<Lhs, Rhs>, const Lhs &, const Rhs &>;

//! @brief The evaluated type of the ABᵀ expression.
template <typename Lhs, typename Rhs>
using ᴀʙᵀ = evaluate<product<Lhs, evaluate<transpose<Rhs>>>>;

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

template <typename Filter> struct conditional_gain {};

template <has_gain Filter> struct conditional_gain<Filter> {
  //! @brief Type of the gain matrix K.
  using gain = Filter::gain;
};

template <typename Filter> struct conditional_innovation {};

template <has_innovation Filter> struct conditional_innovation<Filter> {
  //! @brief Type of the innovation column vector Y.
  using innovation = Filter::innovation;
};

template <typename Filter> struct conditional_innovation_uncertainty {};

template <has_innovation_uncertainty Filter>
struct conditional_innovation_uncertainty<Filter> {
  //! @brief Type of the innovation uncertainty matrix S.
  using innovation_uncertainty = Filter::innovation_uncertainty;
};

template <typename Filter> struct conditional_estimate_uncertainty {};

template <has_estimate_uncertainty Filter>
struct conditional_estimate_uncertainty<Filter> {
  //! @brief Type of the estimated correlated variance matrix P.
  //!
  //! @details Also known as Σ.
  using estimate_uncertainty = Filter::estimate_uncertainty;
};

template <typename Filter> struct conditional_output {};

template <has_output Filter> struct conditional_output<Filter> {
  //! @brief Type of the observation column vector Z.
  //!
  //! @details Also known as Y or O.
  using output = Filter::output;
};

template <typename Filter> struct conditional_state {};

template <has_state Filter> struct conditional_state<Filter> {
  //! @brief Type of the state estimate column vector X.
  using state = Filter::state;
};

// The only way to have a conditional member type is to inherit from a template
// specialization on the member type.
template <typename Filter>
struct conditional_member_types
    : public conditional_estimate_uncertainty<Filter>,
      conditional_gain<Filter>,
      conditional_innovation_uncertainty<Filter>,
      conditional_innovation<Filter>,
      conditional_input_control<Filter>,
      conditional_input<Filter>,
      conditional_state<Filter>,
      conditional_output_model<Filter>,
      conditional_output_uncertainty<Filter>,
      conditional_output<Filter>,
      conditional_prediction_types<Filter>,
      conditional_process_uncertainty<Filter>,
      conditional_state_transition<Filter>,
      conditional_update_types<Filter> {};

//! @}

//! @name Functions
//! @{

//! @brief Compile-time for loop.
//!
//! @details Help compilers with non-type template parameters on members.
template <std::size_t Begin, std::size_t End, std::size_t Increment,
          typename Function>
constexpr void for_constexpr(Function &&function) {
  if constexpr (Begin < End) {
    function(std::integral_constant<std::size_t, Begin>{});
    for_constexpr<Begin + Increment, End, Increment>(
        std::forward<Function>(function));
  }
}

//! @brief Assigns a callable characteristic from its copied initializers.
//!
//! @details The closure type of a lambda expression with captures has a
//! deleted copy assignment operator: such a callable is replaced in place
//! instead. The replacement is constructed before the destruction of the
//! assigned callable, and its non-throwing move construction never leaves the
//! characteristic destroyed. Neither form allocates.
template <typename Callable>
constexpr void assign(Callable &callable, const auto &...values) {
  if constexpr (std::is_move_assignable_v<Callable>) {
    callable = Callable{values...};
  } else {
    static_assert(std::is_nothrow_move_constructible_v<Callable>,
                  "The callable characteristic must be move assignable or "
                  "non-throwing move constructible to be replaced.");

    Callable replacement{values...};
    std::destroy_at(&callable);
    std::construct_at(&callable, std::move(replacement));
  }
}

//! @brief Reads a characteristic of the object.
//!
//! @details A constant reference for an lvalue object: reading does not copy
//! the characteristic, and writing goes through the constrained setters. The
//! moved value for an rvalue object: a reference into the object would dangle
//! past the full-expression.
template <typename Object, typename Characteristic>
constexpr decltype(auto) read(Characteristic &characteristic) {
  if constexpr (std::is_lvalue_reference_v<Object>) {
    return std::as_const(characteristic);
  } else {
    return std::remove_const_t<Characteristic>{std::move(characteristic)};
  }
}

//! @}

//! @name Named Values
//! @{

template <typename Type> struct repacker {
  using type = Type;
};

template <template <typename...> typename Pack, typename... Types>
struct repacker<Pack<Types...>> {
  using type = std::tuple<Types...>;

  static inline constexpr std::size_t size{sizeof...(Types)};
};

template <typename Pack> using repack = repacker<Pack>::type;

//! @brief Size of tuple-like types.
//!
//! @details Convenient short form. In place of `std::tuple_size_v`.
template <typename Pack>
inline constexpr std::size_t size{repacker<Pack>::size};

template <auto Value, auto... Values> struct first_value {
  static constexpr auto value{Value};
};

//! @brief Unpack the first value of the non-type template parameter pack.
template <auto... Values>
inline constexpr auto first_v{first_value<Values...>::value};

template <typename Type> struct not_implemented {
  template <auto Size>
  constexpr explicit not_implemented(
      [[maybe_unused]] const char (&message)[Size]) {
    // The argument message is printed in the compiler error output.
  }

  static constexpr auto type_dependent_false{sizeof(Type) != sizeof(Type)};
  static constexpr auto missing{type_dependent_false};

  static_assert(missing, "This type is not implemented. See compiler message.");
};

//! @brief Constant evaluation concept.
//!
//! @details The value built by the factory is a constant expression.
template <typename Factory>
concept constant_evaluable = requires {
  typename std::bool_constant<(static_cast<void>(Factory{}()), true)>;
};

//! @brief The one matrix.
//!
//! @details User-defined matrix with all its diagonal elements equal
//! to ones, and zeroes everywhere else. This matrix is also known as the
//! identity matrix for square matrices of non-quantity scalar types.
//!
//! @warning A user-defined specialization must be a constant expression,
//! declared `constexpr`. The dynamic initialization of a templated variable is
//! unordered: a filter constructed during the dynamic initialization of another
//! object of static storage duration would copy its zero-initialized storage.
//! Specialize the `ones` factory instead for a value that is not a constant
//! expression.
template <typename Type = double>
inline constexpr Type one{not_implemented<Type>{
    "Implement the linear algebra one-diagonal matrix for this type."}};

//! @brief The singleton one matrix specialization.
template <arithmetic Arithmetic> inline constexpr Arithmetic one<Arithmetic>{1};

//! @brief Linear algebra one matrix specialization point.
//!
//! @details Builds the one matrix at the point of use, never reading a variable
//! of unordered dynamic initialization.
template <typename Type> struct ones {
  [[nodiscard]] static constexpr auto operator()() { return one<Type>; }
};

template <typename Type>
  requires requires { Type::Identity(); }
struct ones<Type> {
  [[nodiscard]] static constexpr auto operator()() { return Type::Identity(); }
};

template <typename Type>
  requires requires { Type::identity(); }
struct ones<Type> {
  [[nodiscard]] static constexpr auto operator()() { return Type::identity(); }
};

//! @brief The constant one matrix specialization.
//!
//! @details Defined only for a backend whose one matrix is a constant
//! expression, which is then constant-initialized.
template <typename Type>
  requires(
              requires { Type::Identity(); } ||
              requires { Type::identity(); }) &&
          constant_evaluable<ones<Type>>
inline constexpr auto one<Type>{ones<Type>{}()};

//! @brief The constant one matrix of a constant-evaluable factory.
//!
//! @details Constant-initialized from the `ones` factory, whichever of its
//! specializations applies, including a user-defined one for a type without a
//! `one` specialization.
template <typename Type>
  requires constant_evaluable<ones<Type>>
inline constexpr auto constant_one{ones<Type>{}()};

//! @brief The one matrix built at the point of use.
//!
//! @details A reference to the constant one matrix when it is a constant
//! expression, so that a filter is constant-initializable and an update binds
//! it without a copy on the stack, otherwise the value built by its backend.
template <typename Type> [[nodiscard]] constexpr decltype(auto) make_one() {
  if constexpr (constant_evaluable<ones<Type>>) {
    // A cast: GCC deduces a value from a parenthesized variable template.
    return static_cast<const decltype(constant_one<Type>) &>(
        constant_one<Type>);
  } else {
    return ones<Type>{}();
  }
}

//! @brief The zero matrix.
//!
//! @details User-defined.
//!
//! @warning A user-defined specialization must be a constant expression,
//! declared `constexpr`. The dynamic initialization of a templated variable is
//! unordered: a filter constructed during the dynamic initialization of another
//! object of static storage duration would copy its zero-initialized storage.
//! Specialize the `zeros` factory instead for a value that is not a constant
//! expression.
template <typename Type = double>
inline constexpr Type zero{kalman_filter::internal::not_implemented<Type>{
    "Implement the linear algebra zero matrix for this type."}};

//! @brief The singleton zero matrix specialization.
template <arithmetic Arithmetic>
inline constexpr Arithmetic zero<Arithmetic>{0};

//! @brief Linear algebra zero matrix specialization point.
//!
//! @details Builds the zero matrix at the point of use, never reading a
//! variable of unordered dynamic initialization.
template <typename Type> struct zeros {
  [[nodiscard]] static constexpr auto operator()() { return zero<Type>; }
};

template <typename Type>
  requires requires { Type::Zero(); }
struct zeros<Type> {
  [[nodiscard]] static constexpr auto operator()() { return Type::Zero(); }
};

template <typename Type>
  requires requires { Type::zero(); }
struct zeros<Type> {
  [[nodiscard]] static constexpr auto operator()() { return Type::zero(); }
};

//! @brief The constant zero matrix specialization.
//!
//! @details Defined only for a backend whose zero matrix is a constant
//! expression, which is then constant-initialized.
template <typename Type>
  requires(
              requires { Type::Zero(); } || requires { Type::zero(); }) &&
          constant_evaluable<zeros<Type>>
inline constexpr auto zero<Type>{zeros<Type>{}()};

//! @brief The constant zero matrix of a constant-evaluable factory.
//!
//! @details Constant-initialized from the `zeros` factory, whichever of its
//! specializations applies, including a user-defined one for a type without a
//! `zero` specialization.
template <typename Type>
  requires constant_evaluable<zeros<Type>>
inline constexpr auto constant_zero{zeros<Type>{}()};

//! @brief The zero matrix built at the point of use.
//!
//! @details A reference to the constant zero matrix when it is a constant
//! expression, so that a filter is constant-initializable and an update binds
//! it without a copy on the stack, otherwise the value built by its backend.
template <typename Type> [[nodiscard]] constexpr decltype(auto) make_zero() {
  if constexpr (constant_evaluable<zeros<Type>>) {
    // A cast: GCC deduces a value from a parenthesized variable template.
    return static_cast<const decltype(constant_zero<Type>) &>(
        constant_zero<Type>);
  } else {
    return zeros<Type>{}();
  }
}

template <typename Callable> struct scope_exit {
  Callable callable;
  constexpr ~scope_exit() { callable(); }
};

//! @}

} // namespace fcarouge::kalman_filter::internal

#endif // FCAROUGE_KALMAN_FILTER_INTERNAL_UTILITY_HPP
