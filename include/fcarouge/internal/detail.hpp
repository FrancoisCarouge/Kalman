/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter for C++
Version 0.2.0
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

#ifndef FCAROUGE_INTERNAL_DETAIL_HPP
#define FCAROUGE_INTERNAL_DETAIL_HPP

namespace fcarouge {

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned state estimate column vector X is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::x() const
    -> const state & {
  return filter.x;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned state estimate column vector X is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::x()
    -> state & {
  return filter.x;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::x(
    const auto &value, const auto &...values) {
  filter.x = std::move(state{value, values...});
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned observation column vector Z is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::z() const
    -> const output & {
  return filter.z;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned control column vector U is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::u() const
    -> const input &requires(not std::is_same_v<Input, void>) {
                      return filter.u;
                    }

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned estimated covariance matrix P is unexpectedly "
            "discarded.")]] inline constexpr auto kalman<State, Output, Input,
                                                         Divide, UpdateTypes,
                                                         PredictionTypes>::p()
    const -> const estimate_uncertainty & {
  return filter.p;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned estimated covariance matrix P is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::p()
    -> estimate_uncertainty & {
  return filter.p;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::p(
    const auto &value, const auto &...values) {
  filter.p = std::move(estimate_uncertainty{value, values...});
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned process noise covariance matrix Q is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::q() const
    -> const process_uncertainty & {
  return filter.q;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned process noise covariance matrix Q is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::q()
    -> process_uncertainty & {
  return filter.q;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::q(
    const auto &value, const auto &...values) {
  if constexpr (std::is_convertible_v<decltype(value), process_uncertainty>) {
    filter.q = std::move(process_uncertainty{value, values...});
  } else {
    using noise_process_function = decltype(filter.noise_process_q);
    filter.noise_process_q =
        std::move(noise_process_function{value, values...});
  }
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned observation noise covariance matrix R is "
            "unexpectedly discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::r() const
    -> const output_uncertainty & {
  return filter.r;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned observation noise covariance matrix R is "
            "unexpectedly discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::r()
    -> output_uncertainty & {
  return filter.r;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::r(
    const auto &value, const auto &...values) {
  if constexpr (std::is_convertible_v<decltype(value), output_uncertainty>) {
    filter.r = std::move(output_uncertainty{value, values...});
  } else {
    using noise_observation_function = decltype(filter.noise_observation_r);
    filter.noise_observation_r =
        std::move(noise_observation_function{value, values...});
  }
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned state transition matrix F is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::f() const
    -> const state_transition & {
  return filter.f;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned state transition matrix F is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::f()
    -> state_transition & {
  return filter.f;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::f(
    const auto &value, const auto &...values) {
  if constexpr (std::is_convertible_v<decltype(value), state_transition>) {
    filter.f = std::move(state_transition{value, values...});
  } else {
    using transition_state_function = decltype(filter.transition_state_f);
    filter.transition_state_f =
        std::move(transition_state_function{value, values...});
  }
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned observation transition matrix H is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::h() const
    -> const output_model & {
  return filter.h;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned observation transition matrix H is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::h()
    -> output_model & {
  return filter.h;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::h(
    const auto &value, const auto &...values) {
  if constexpr (std::is_convertible_v<decltype(value), output_model>) {
    filter.h = std::move(output_model{value, values...});
  } else {
    using observation_state_function = decltype(filter.observation_state_h);
    filter.observation_state_h =
        std::move(observation_state_function{value, values...});
  }
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned control transition matrix G is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::g() const
    -> const input_control &requires(not std::is_same_v<Input, void>) {
                              return filter.g;
                            }

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned control transition matrix G is unexpectedly "
            "discarded.")]] inline constexpr auto kalman<State, Output, Input,
                                                         Divide, UpdateTypes,
                                                         PredictionTypes>::g()
    -> input_control &requires(not std::is_same_v<Input, void>) {
                        return filter.g;
                      }

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void kalman<State, Output, Input, Divide, UpdateTypes,
                             PredictionTypes>::g(const auto &value,
                                                 const auto &...values)
  requires(not std::is_same_v<Input, void>)
{
  if constexpr (std::is_convertible_v<decltype(value), input_control>) {
    filter.g = std::move(input_control{value, values...});
  } else {
    using transition_control_function = decltype(filter.transition_control_g);
    filter.transition_control_g =
        std::move(transition_control_function{value, values...});
  }
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned gain matrix K is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::k() const
    -> const gain & {
  return filter.k;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned innovation column vector Y is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::y() const
    -> const innovation & {
  return filter.y;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
[[nodiscard("The returned innovation uncertainty matrix S is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::s() const
    -> const innovation_uncertainty & {
  return filter.s;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::transition(
    const auto &callable) {
  filter.transition = callable;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::observation(
    const auto &callable) {
  filter.observation = callable;
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::update(
    const auto &...arguments) {
  filter.update(arguments...);
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
template <std::size_t Position>
[[nodiscard("The returned update argument is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::update()
    const {
  return std::get<Position>(filter.update_arguments);
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
inline constexpr void
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::predict(
    const auto &...arguments) {
  filter.predict(arguments...);
}

template <typename State, typename Output, typename Input, typename Divide,
          typename UpdateTypes, typename PredictionTypes>
template <std::size_t Position>
[[nodiscard("The returned prediction argument is unexpectedly "
            "discarded.")]] inline constexpr auto
kalman<State, Output, Input, Divide, UpdateTypes, PredictionTypes>::predict()
    const {
  return std::get<Position>(filter.prediction_arguments);
}

} // namespace fcarouge

#endif // FCAROUGE_INTERNAL_DETAIL_HPP
