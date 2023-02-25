/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
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

#ifndef FCAROUGE_INTERNAL_PREDICT_TPP
#define FCAROUGE_INTERNAL_PREDICT_TPP

namespace fcarouge {

template <typename State, typename Input, typename... PredictionTypes>
inline constexpr predict<State, Input, PredictionTypes...>::predict(
    [[maybe_unused]] state &x, [[maybe_unused]] estimate_uncertainty &p)
    : model{x, p} {}

template <typename State, typename Input, typename... PredictionTypes>
[[nodiscard("The returned state estimate column vector X is unexpectedly "
            "discarded.")]] inline constexpr auto
predict<State, Input, PredictionTypes...>::x() const -> const state & {
  return model.x;
}

template <typename State, typename Input, typename... PredictionTypes>
[[nodiscard("The returned state estimate column vector X is unexpectedly "
            "discarded.")]] inline constexpr auto
predict<State, Input, PredictionTypes...>::x() -> state & {
  return model.x;
}

template <typename State, typename Input, typename... PredictionTypes>
[[nodiscard("The returned control column vector U is unexpectedly "
            "discarded.")]] inline constexpr auto
predict<State, Input, PredictionTypes...>::u() const
    -> const input &requires(not std::is_same_v<Input, void>) {
                      return model.u;
                    }

template <typename State, typename Input, typename... PredictionTypes>
[[nodiscard(
    "The returned estimated covariance matrix P is unexpectedly "
    "discarded.")]] inline constexpr auto predict<State, Input,
                                                  PredictionTypes...>::p() const
    -> const estimate_uncertainty & {
  return model.p;
}

template <typename State, typename Input, typename... PredictionTypes>
[[nodiscard("The returned estimated covariance matrix P is unexpectedly "
            "discarded.")]] inline constexpr auto
predict<State, Input, PredictionTypes...>::p() -> estimate_uncertainty & {
  return model.p;
}

template <typename State, typename Input, typename... PredictionTypes>
[[nodiscard("The returned process noise covariance matrix Q is unexpectedly "
            "discarded.")]] inline constexpr auto
predict<State, Input, PredictionTypes...>::q() const
    -> const process_uncertainty & {
  return model.q;
}

template <typename State, typename Input, typename... PredictionTypes>
[[nodiscard("The returned process noise covariance matrix Q is unexpectedly "
            "discarded.")]] inline constexpr auto
predict<State, Input, PredictionTypes...>::q() -> process_uncertainty & {
  return model.q;
}

template <typename State, typename Input, typename... PredictionTypes>
inline constexpr void
predict<State, Input, PredictionTypes...>::q(const auto &value,
                                             const auto &...values) {
  if constexpr (std::is_convertible_v<decltype(value), process_uncertainty>) {
    model.q = std::move(process_uncertainty{value, values...});
  } else {
    using noise_process_function = decltype(model.noise_process_q);
    model.noise_process_q = std::move(noise_process_function{value, values...});
  }
}

template <typename State, typename Input, typename... PredictionTypes>
[[nodiscard("The returned state transition matrix F is unexpectedly "
            "discarded.")]] inline constexpr auto
predict<State, Input, PredictionTypes...>::f() const
    -> const state_transition & {
  return model.f;
}

template <typename State, typename Input, typename... PredictionTypes>
[[nodiscard("The returned state transition matrix F is unexpectedly "
            "discarded.")]] inline constexpr auto
predict<State, Input, PredictionTypes...>::f() -> state_transition & {
  return model.f;
}

template <typename State, typename Input, typename... PredictionTypes>
inline constexpr void
predict<State, Input, PredictionTypes...>::f(const auto &value,
                                             const auto &...values) {
  if constexpr (std::is_convertible_v<decltype(value), state_transition>) {
    model.f = std::move(state_transition{value, values...});
  } else {
    using transition_state_function = decltype(model.transition_state_f);
    model.transition_state_f =
        std::move(transition_state_function{value, values...});
  }
}

template <typename State, typename Input, typename... PredictionTypes>
[[nodiscard("The returned control transition matrix G is unexpectedly "
            "discarded.")]] inline constexpr auto
predict<State, Input, PredictionTypes...>::g() const
    -> const input_control &requires(not std::is_same_v<Input, void>) {
                              return model.g;
                            }

template <typename State, typename Input, typename... PredictionTypes>
[[nodiscard(
    "The returned control transition matrix G is unexpectedly "
    "discarded.")]] inline constexpr auto predict<State, Input,
                                                  PredictionTypes...>::g()
    -> input_control &requires(not std::is_same_v<Input, void>) {
                        return model.g;
                      }

template <typename State, typename Input, typename... PredictionTypes>
inline constexpr void predict<State, Input, PredictionTypes...>::g(
    const auto &value, const auto &...values)
  requires(not std::is_same_v<Input, void>)
{
  if constexpr (std::is_convertible_v<decltype(value), input_control>) {
    model.g = std::move(input_control{value, values...});
  } else {
    using transition_control_function = decltype(model.transition_control_g);
    model.transition_control_g =
        std::move(transition_control_function{value, values...});
  }
}

template <typename State, typename Input, typename... PredictionTypes>
inline constexpr void
predict<State, Input, PredictionTypes...>::transition(const auto &callable) {
  model.transition = callable;
}

template <typename State, typename Input, typename... PredictionTypes>
inline constexpr void predict<State, Input, PredictionTypes...>::operator()(
    const auto &...arguments) {
  model(arguments...);
}

template <typename State, typename Input, typename... PredictionTypes>
template <std::size_t Position>
[[nodiscard("The returned update argument is unexpectedly "
            "discarded.")]] inline constexpr auto
predict<State, Input, PredictionTypes...>::operator()() const {
  return std::get<Position>(model.prediction_arguments);
}

} // namespace fcarouge

#endif // FCAROUGE_INTERNAL_PREDICT_TPP
