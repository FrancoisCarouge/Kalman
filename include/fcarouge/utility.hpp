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

#ifndef FCAROUGE_UTILITY_HPP
#define FCAROUGE_UTILITY_HPP

//! @file
//! @brief The collection of utilities supporting the library
//!
//! @details Definitions and documentation of supporting concepts and types.

#include "internal/utility.hpp"

#include <cstddef>
#include <utility>

namespace fcarouge {
//! @name Concepts
//! @{

//! @brief Kalman filter concept.
//!
//! @details This library's Kalman filters.
template <typename Type>
concept kalman_filter = internal::kalman_filter<Type>;

//! @brief Arithmetic concept.
//!
//! @details Any integer or floating point type.
template <typename Type>
concept arithmetic = internal::arithmetic<Type>;

//! @brief Algebraic concept.
//!
//! @details Not an arithmetic type.
template <typename Type>
concept algebraic = internal::algebraic<Type>;

//! @brief Eigen3 algebraic concept.
//!
//! @details A third party Eigen3 algebraic concept.
template <typename Type>
concept eigen = internal::eigen<Type>;

//! @brief Filter input support concept.
//!
//! @details The filter supports the input related functionality: `input` type
//! member and `u()` method.
template <typename Filter>
concept has_input = internal::has_input<Filter>;

//! @brief Filter process uncertainty support concept.
//!
//! @details The filter supports the process uncertainty related functionality:
//! `process_uncertainty` type member and `q()` method.
template <typename Filter>
concept has_process_uncertainty = internal::has_process_uncertainty<Filter>;

//! @brief Filter output uncertainty support concept.
//!
//! @details The filter supports the output uncertainty related functionality:
//! `output_uncertainty` type member and `r()` method.
template <typename Filter>
concept has_output_uncertainty = internal::has_output_uncertainty<Filter>;

//! @brief Filter prediction pack support concept.
//!
//! @details The filter supports the prediction parameters related
//! functionality: `prediction_types` type member and parameters for the
//! `predict()` method.
template <typename Filter>
concept has_prediction_types = internal::has_prediction_types<Filter>;

//! @brief Filter input control support concept.
//!
//! @details The filter supports the input control related functionality:
//! `input_control` type member and `g()` method.
template <typename Filter>
concept has_input_control = internal::has_input_control<Filter>;

//! @brief Filter state transition support concept.
//!
//! @details The filter supports the state transition related functionality:
//! `state_transition` type member and `f()` method.
template <typename Filter>
concept has_state_transition = internal::has_state_transition<Filter>;

//! @brief Filter update pack support concept.
//!
//! @details The filter supports the update parameters related functionality:
//! `update_types` type member and parameters for the `update()` method.
template <typename Filter>
concept has_update_types = internal::has_update_types<Filter>;

//! @brief Filter output model support concept.
//!
//! @details The filter supports the output model related functionality:
//! `output_model` type member and `h()` method.
template <typename Filter>
concept has_output_model = internal::has_output_model<Filter>;

//! @}

//! @name Types
//! @{

//! @brief Type of the empty tuple.
//!
//! @details A tuple with no `pack` types.
using empty_tuple = internal::empty_tuple;

//! @brief Unpack the first type of the type template parameter pack.
//!
//! @details Shorthand for `std::tuple_element_t<0, std::tuple<Types...>>`.
template <typename... Types> using first = internal::first<Types...>;

//! @brief The deduced result type of the product.
template <typename Lhs, typename Rhs>
using product = internal::product<Lhs, Rhs>;

//! @brief The evaluated type of the ABᵀ expression.
template <typename Numerator, typename Denominator>
using ᴀʙᵀ = internal::ᴀʙᵀ<Numerator, Denominator>;

//! @}

//! @name Functions
//! @{

//! @brief Compile-time for loop.
//!
//! @details Help compilers with non-type template parameters on members.
template <std::size_t Begin, std::size_t End, std::size_t Increment,
          typename Function>
inline constexpr void for_constexpr(Function &&function) {
  internal::for_constexpr<Begin, End, Increment>(
      std::forward<Function>(function));
}

//! @brief A user-definable algebraic division solution.
//!
//! @details Matrix division is a mathematical abuse of terminology. Informally
//! defined as multiplication by the inverse. Similarly to division by zero in
//! real numbers, there exists matrices that are not invertible. Remember the
//! division operation is not commutative. Matrix inversion can be avoided by
//! solving `X * rhs = lhs` for `rhs` through a decomposer. There exists several
//! ways to decompose and solve the equation. Implementations trade off
//! numerical stability, triangularity, symmetry, space, time, etc. Dividing an
//! `R1 x C` matrix by an `R2 x C` matrix results in an `R1 x R2` matrix.
template <typename Numerator, algebraic Denominator>
constexpr auto operator/(const Numerator &lhs,
                         const Denominator &rhs) -> ᴀʙᵀ<Numerator, Denominator>;

//! @}

//! @name Named Values
//! @{

//! @brief Unpack the first value of the non-type template parameter pack.
template <auto... Values>
inline constexpr auto first_v{internal::first_v<Values...>};

//! @brief The identity matrix.
//!
//! @details User-defined.
template <typename Type = double>
inline constexpr Type identity{internal::not_implemented<Type>{
    "Implement the linear algebra identity matrix for this type."}};

//! @brief The singleton identity matrix specialization.
template <arithmetic Arithmetic>
inline constexpr Arithmetic identity<Arithmetic>{1};

template <typename Type>
  requires requires { Type::Identity(); }
inline auto identity<Type>{Type::Identity()};

template <typename Type>
  requires requires { Type::identity(); }
inline auto identity<Type>{Type::identity()};

//! @brief The zero matrix.
//!
//! @details User-defined.
template <typename Type = double>
inline constexpr Type zero{internal::not_implemented<Type>{
    "Implement the linear algebra zero matrix for this type."}};

//! @brief The singleton zero matrix specialization.
template <arithmetic Arithmetic>
inline constexpr Arithmetic zero<Arithmetic>{0};

template <typename Type>
  requires requires { Type::Zero(); }
inline auto zero<Type>{Type::Zero()};

template <typename Type>
  requires requires { Type::zero(); }
inline auto zero<Type>{Type::zero()};

//! @}
} // namespace fcarouge

#endif // FCAROUGE_UTILITY_HPP
