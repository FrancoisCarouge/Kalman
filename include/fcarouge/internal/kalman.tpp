/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.1
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

#include <tuple>
#include <type_traits>
#include <utility>

namespace fcarouge {
template <typename Filter>
template <typename... Arguments>
inline constexpr kalman<Filter>::kalman(Arguments... arguments)
    : filter{internal::filter<Filter>(arguments...)} {}

template <typename Filter>
[[nodiscard(
    "The returned state estimate column vector X is unexpectedly discarded.")]]
inline constexpr auto &&kalman<Filter>::x(this auto &&self) {
  return std::forward<decltype(self)>(self).filter.x;
}

template <typename Filter>
inline constexpr void kalman<Filter>::x(const auto &value,
                                        const auto &...values) {
  filter.x = std::move(state{value, values...});
}

template <typename Filter>
[[nodiscard("The returned observation column vector Z is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Filter>::z() const -> const output & {
  return filter.z;
}

template <typename Filter>
[[nodiscard("The returned control column vector U is unexpectedly "
            "discarded.")]] inline constexpr const auto &
kalman<Filter>::u() const
  requires(has_input<Filter>)
{
  return filter.u;
}

template <typename Filter>
[[nodiscard("The returned estimated covariance matrix P is unexpectedly "
            "discarded.")]] inline constexpr auto &&
kalman<Filter>::p(this auto &&self) {
  return std::forward<decltype(self)>(self).filter.p;
}

template <typename Filter>
inline constexpr void kalman<Filter>::p(const auto &value,
                                        const auto &...values) {
  filter.p = std::move(estimate_uncertainty{value, values...});
}

template <typename Filter>
[[nodiscard("The returned process noise covariance matrix Q is unexpectedly "
            "discarded.")]] inline constexpr auto &&
kalman<Filter>::q(this auto &&self)
  requires(has_process_uncertainty<Filter>)
{
  return std::forward<decltype(self)>(self).filter.q;
}

template <typename Filter>
inline constexpr void kalman<Filter>::q(const auto &value,
                                        const auto &...values)
  requires(has_process_uncertainty<Filter>)
{
  if constexpr (std::is_convertible_v<decltype(value),
                                      typename Filter::process_uncertainty>) {
    filter.q =
        std::move(typename Filter::process_uncertainty{value, values...});
  } else {
    using noise_process_function = decltype(filter.noise_process_q);
    filter.noise_process_q =
        std::move(noise_process_function{value, values...});
  }
}

template <typename Filter>
[[nodiscard("The returned observation noise covariance matrix R is "
            "unexpectedly discarded.")]] inline constexpr auto &&
kalman<Filter>::r(this auto &&self)
  requires(has_output_uncertainty<Filter>)
{
  return std::forward<decltype(self)>(self).filter.r;
}

template <typename Filter>
inline constexpr void kalman<Filter>::r(const auto &value,
                                        const auto &...values)
  requires(has_output_uncertainty<Filter>)
{
  if constexpr (std::is_convertible_v<decltype(value),
                                      typename Filter::output_uncertainty>) {
    filter.r = std::move(typename Filter::output_uncertainty{value, values...});
  } else {
    using noise_observation_function = decltype(filter.noise_observation_r);
    filter.noise_observation_r =
        std::move(noise_observation_function{value, values...});
  }
}

template <typename Filter>
[[nodiscard("The returned state transition matrix F is unexpectedly "
            "discarded.")]] inline constexpr auto &&
kalman<Filter>::f(this auto &&self)
  requires(has_state_transition<Filter>)
{
  return std::forward<decltype(self)>(self).filter.f;
}

template <typename Filter>
inline constexpr void kalman<Filter>::f(const auto &value,
                                        const auto &...values)
  requires(has_state_transition<Filter>)
{
  if constexpr (std::is_convertible_v<decltype(value),
                                      typename Filter::state_transition>) {
    filter.f = std::move(typename Filter::state_transition{value, values...});
  } else {
    using transition_state_function = decltype(filter.transition_state_f);
    filter.transition_state_f =
        std::move(transition_state_function{value, values...});
  }
}

template <typename Filter>
[[nodiscard("The returned observation transition matrix H is unexpectedly "
            "discarded.")]] inline constexpr auto &&
kalman<Filter>::h(this auto &&self)
  requires(has_output_model<Filter>)
{
  return std::forward<decltype(self)>(self).filter.h;
}

template <typename Filter>
inline constexpr void kalman<Filter>::h(const auto &value,
                                        const auto &...values)
  requires(has_output_model<Filter>)
{
  if constexpr (std::is_convertible_v<decltype(value),
                                      typename Filter::output_model>) {
    filter.h = std::move(typename Filter::output_model{value, values...});
  } else {
    using observation_state_function = decltype(filter.observation_state_h);
    filter.observation_state_h =
        std::move(observation_state_function{value, values...});
  }
}

template <typename Filter>
[[nodiscard("The returned control transition matrix G is unexpectedly "
            "discarded.")]] inline constexpr auto &&
kalman<Filter>::g(this auto &&self)
  requires(has_input_control<Filter>)
{
  return std::forward<decltype(self)>(self).filter.g;
}

template <typename Filter>
inline constexpr void kalman<Filter>::g(const auto &value,
                                        const auto &...values)
  requires(has_input_control<Filter>)
{
  using transition_control_function = decltype(filter.transition_control_g);
  filter.transition_control_g =
      std::move(transition_control_function{value, values...});
}

template <typename Filter>
[[nodiscard("The returned gain matrix K is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Filter>::k() const -> const gain & {
  return filter.k;
}

template <typename Filter>
[[nodiscard("The returned innovation column vector Y is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Filter>::y() const -> const innovation & {
  return filter.y;
}

template <typename Filter>
[[nodiscard("The returned innovation uncertainty matrix S is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Filter>::s() const -> const innovation_uncertainty & {
  return filter.s;
}

template <typename Filter>
inline constexpr void kalman<Filter>::transition(const auto &callable) {
  filter.transition = callable;
}

template <typename Filter>
inline constexpr void kalman<Filter>::observation(const auto &callable) {
  filter.observation = callable;
}

template <typename Filter>
inline constexpr void kalman<Filter>::update(const auto &...arguments) {
  filter.update(arguments...);
}

template <typename InternalFilter>
template <auto Position>
[[nodiscard("The returned update argument is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<InternalFilter>::update() const {
  return std::get<Position>(filter.update_arguments);
}

template <typename Filter>
inline constexpr void kalman<Filter>::predict(const auto &...arguments) {
  filter.predict(arguments...);
}

template <typename Filter>
template <auto Position>
[[nodiscard("The returned prediction argument is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<Filter>::predict() const {
  return std::get<Position>(filter.prediction_arguments);
}
} // namespace fcarouge

#endif // FCAROUGE_INTERNAL_KALMAN_TPP
