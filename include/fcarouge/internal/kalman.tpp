/*_  __          _      __  __          _   _
 | |/ /    /\   | |    |  \/  |   /\   | \ | |
 | ' /    /  \  | |    | \  / |  /  \  |  \| |
 |  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
 | . \  / ____ \| |____| |  | |/ ____ \| |\  |
 |_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter for C++
Version 0.1.0
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

#ifndef FCAROUGE_INTERNAL_KALMAN_TPP
#define FCAROUGE_INTERNAL_KALMAN_TPP

#include "fcarouge/kalman.hpp"

#include <cstddef>

namespace fcarouge
{

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
[[nodiscard("The returned state estimate column vector X is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::x() const -> state
{
  return filter.x;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::x(const state &value)
{
  filter.x = value;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::x(state &&value)
{
  filter.x = std::move(value);
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::x(const value_type &value,
                                        const std::same_as<value_type> auto
                                            &...values)
{
  filter.x = std::move(state{ value, values... });
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::x(value_type &&value,
                                        std::same_as<value_type> auto
                                            &&...values)
{
  filter.x = std::move(state{ std::forward<decltype(value)>(value),
                              std::forward<decltype(values)>(values)... });
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
[[nodiscard("The returned observation column vector Z is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::z() const -> output
{
  return filter.z;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
[[nodiscard("The returned control column vector U is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::u() const -> input requires(Input > 0)
{
  return filter.u;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
[[nodiscard("The returned estimated covariance matrix P is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::p() const -> estimate_uncertainty
{
  return filter.p;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::p(const estimate_uncertainty &value)
{
  filter.p = value;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::p(estimate_uncertainty &&value)
{
  filter.p = std::move(value);
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::p(const value_type &value,
                                        const std::same_as<value_type> auto
                                            &...values)
{
  filter.p = std::move(estimate_uncertainty{ value, values... });
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::p(value_type &&value,
                                        std::same_as<value_type> auto
                                            &&...values)
{
  filter.p = std::move(
      estimate_uncertainty{ std::forward<decltype(value)>(value),
                            std::forward<decltype(values)>(values)... });
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
[[nodiscard("The returned process noise covariance matrix Q is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::q() const -> process_uncertainty
{
  return filter.q;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::q(const process_uncertainty &value)
{
  filter.q = value;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::q(process_uncertainty &&value)
{
  filter.q = std::move(value);
}

//! @todo Reset functions or values when the other is set?
template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::q(const value_type &value,
                                        const std::same_as<value_type> auto
                                            &...values)
{
  filter.q = std::move(process_uncertainty{ value, values... });
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::q(value_type &&value,
                                        std::same_as<value_type> auto
                                            &&...values)
{
  filter.q = std::move(
      process_uncertainty{ std::forward<decltype(value)>(value),
                           std::forward<decltype(values)>(values)... });
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::q(const noise_process_function &callable)
{
  filter.noise_process_q = callable;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::q(noise_process_function &&callable)
{
  filter.noise_process_q = std::forward<decltype(callable)>(callable);
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
[[nodiscard("The returned observation noise covariance matrix R is "
            "unexpectedly discarded.")]] inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::r() const -> output_uncertainty
{
  return filter.r;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::r(const output_uncertainty &value)
{
  filter.r = value;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::r(output_uncertainty &&value)
{
  filter.r = std::move(value);
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::r(const value_type &value,
                                        const std::same_as<value_type> auto
                                            &...values)
{
  filter.r = std::move(output_uncertainty{ value, values... });
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::r(value_type &&value,
                                        std::same_as<value_type> auto
                                            &&...values)
{
  filter.r = std::move(
      output_uncertainty{ std::forward<decltype(value)>(value),
                          std::forward<decltype(values)>(values)... });
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::r(const noise_observation_function
                                            &callable)
{
  filter.noise_observation_r = callable;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::r(noise_observation_function &&callable)
{
  filter.noise_observation_r = std::forward<decltype(callable)>(callable);
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
[[nodiscard("The returned state transition matrix F is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::f() const -> state_transition
{
  return filter.f;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::f(const state_transition &value)
{
  filter.f = value;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::f(state_transition &&value)
{
  filter.f = std::move(value);
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::f(const value_type &value,
                                        const std::same_as<value_type> auto
                                            &...values)
{
  filter.f = std::move(state_transition{ value, values... });
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::f(value_type &&value,
                                        std::same_as<value_type> auto
                                            &&...values)
{
  filter.f =
      std::move(state_transition{ std::forward<decltype(value)>(value),
                                  std::forward<decltype(values)>(values)... });
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::f(const transition_state_function
                                            &callable)
{
  filter.transition_state_f = callable;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::f(transition_state_function &&callable)
{
  filter.transition_state_f = std::forward<decltype(callable)>(callable);
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
[[nodiscard("The returned observation transition matrix H is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::h() const -> output_model
{
  return filter.h;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::h(const output_model &value)
{
  filter.h = value;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::h(output_model &&value)
{
  filter.h = std::move(value);
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::h(const value_type &value,
                                        const std::same_as<value_type> auto
                                            &...values)
{
  filter.h = std::move(output_model{ value, values... });
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::h(value_type &&value,
                                        std::same_as<value_type> auto
                                            &&...values)
{
  filter.h =
      std::move(output_model{ std::forward<decltype(value)>(value),
                              std::forward<decltype(values)>(values)... });
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::h(const observation_state_function
                                            &callable)
{
  filter.observation_state_h = callable;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::h(observation_state_function &&callable)
{
  filter.observation_state_h = std::forward<decltype(callable)>(callable);
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
[[nodiscard("The returned control transition matrix G is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::g() const -> input_control
    requires(Input > 0)
{
  return filter.g;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::g(const input_control
                                            &value) requires(Input > 0)
{
  filter.g = value;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::g(input_control &&value) requires(Input >
                                                                        0)
{
  filter.g = std::move(value);
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::g(const value_type &value,
                                        const std::same_as<value_type> auto
                                            &...values)
{
  filter.g = std::move(input_control{ value, values... });
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::g(value_type &&value,
                                        std::same_as<value_type> auto
                                            &&...values)
{
  filter.g =
      std::move(input_control{ std::forward<decltype(value)>(value),
                               std::forward<decltype(values)>(values)... });
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::g(const transition_control_function
                                            &callable)
{
  filter.transition_control_g = callable;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::g(transition_control_function &&callable)
{
  filter.transition_control_g = std::forward<decltype(callable)>(callable);
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
[[nodiscard("The returned gain matrix K is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::k() const -> gain
{
  return filter.k;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
[[nodiscard("The returned innovation column vector Y is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::y() const -> innovation
{
  return filter.y;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
[[nodiscard("The returned innovation uncertainty matrix S is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::s() const -> innovation_uncertainty
{
  return filter.s;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::transition(const transition_function
                                                     &callable)
{
  filter.transition = callable;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::transition(transition_function &&callable)
{
  filter.transition = std::forward<decltype(callable)>(callable);
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::observation(const observation_function
                                                      &callable)
{
  filter.observation = callable;
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::observation(observation_function
                                                      &&callable)
{
  filter.observation = std::forward<decltype(callable)>(callable);
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
template <typename... InputTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::operator()(const auto &...arguments)
{
  filter.template operator()<InputTypes...>(arguments...);
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::update(const auto &...arguments)
{
  filter.update(arguments...);
}

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename UpdateTypes,
          typename PredictionTypes>
inline constexpr void
kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide, Identity,
       UpdateTypes, PredictionTypes>::predict(const auto &...arguments)
{
  filter.predict(arguments...);
}

} // namespace fcarouge

#endif // FCAROUGE_INTERNAL_KALMAN_TPP
