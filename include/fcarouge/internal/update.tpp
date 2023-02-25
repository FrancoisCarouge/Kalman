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

#ifndef FCAROUGE_INTERNAL_UPDATE_TPP
#define FCAROUGE_INTERNAL_UPDATE_TPP

namespace fcarouge {

template <typename State, typename Output, typename... UpdateTypes>
inline constexpr update<State, Output, UpdateTypes...>::update(
    state &x, estimate_uncertainty &p)
    : model{x, p} {}

template <typename State, typename Output, typename... UpdateTypes>
[[nodiscard("The returned state estimate column vector X is unexpectedly "
            "discarded.")]] inline constexpr auto
update<State, Output, UpdateTypes...>::x() const -> const state & {
  return model.x;
}

template <typename State, typename Output, typename... UpdateTypes>
[[nodiscard("The returned state estimate column vector X is unexpectedly "
            "discarded.")]] inline constexpr auto
update<State, Output, UpdateTypes...>::x() -> state & {
  return model.x;
}

template <typename State, typename Output, typename... UpdateTypes>
[[nodiscard("The returned observation column vector Z is unexpectedly "
            "discarded.")]] inline constexpr auto
update<State, Output, UpdateTypes...>::z() const -> const output & {
  return model.z;
}

template <typename State, typename Output, typename... UpdateTypes>
[[nodiscard("The returned estimated covariance matrix P is unexpectedly "
            "discarded.")]] inline constexpr auto
update<State, Output, UpdateTypes...>::p() const
    -> const estimate_uncertainty & {
  return model.p;
}

template <typename State, typename Output, typename... UpdateTypes>
[[nodiscard("The returned estimated covariance matrix P is unexpectedly "
            "discarded.")]] inline constexpr auto
update<State, Output, UpdateTypes...>::p() -> estimate_uncertainty & {
  return model.p;
}

template <typename State, typename Output, typename... UpdateTypes>
[[nodiscard("The returned observation transition matrix H is unexpectedly "
            "discarded.")]] inline constexpr auto
update<State, Output, UpdateTypes...>::h() const -> const output_model & {
  return model.h;
}

template <typename State, typename Output, typename... UpdateTypes>
[[nodiscard("The returned observation transition matrix H is unexpectedly "
            "discarded.")]] inline constexpr auto
update<State, Output, UpdateTypes...>::h() -> output_model & {
  return model.h;
}

template <typename State, typename Output, typename... UpdateTypes>
inline constexpr void
update<State, Output, UpdateTypes...>::h(const auto &value,
                                         const auto &...values) {
  if constexpr (std::is_convertible_v<decltype(value), output_model>) {
    model.h = std::move(output_model{value, values...});
  } else {
    using observation_state_function = decltype(model.observation_state_h);
    model.observation_state_h =
        std::move(observation_state_function{value, values...});
  }
}

template <typename State, typename Output, typename... UpdateTypes>
[[nodiscard("The returned observation noise covariance matrix R is "
            "unexpectedly discarded.")]] inline constexpr auto
update<State, Output, UpdateTypes...>::r() const -> const output_uncertainty & {
  return model.r;
}

template <typename State, typename Output, typename... UpdateTypes>
[[nodiscard("The returned observation noise covariance matrix R is "
            "unexpectedly discarded.")]] inline constexpr auto
update<State, Output, UpdateTypes...>::r() -> output_uncertainty & {
  return model.r;
}

template <typename State, typename Output, typename... UpdateTypes>
inline constexpr void
update<State, Output, UpdateTypes...>::r(const auto &value,
                                         const auto &...values) {
  if constexpr (std::is_convertible_v<decltype(value), output_uncertainty>) {
    model.r = std::move(output_uncertainty{value, values...});
  } else {
    using noise_observation_function = decltype(model.noise_observation_r);
    model.noise_observation_r =
        std::move(noise_observation_function{value, values...});
  }
}

template <typename State, typename Output, typename... UpdateTypes>
[[nodiscard("The returned gain matrix K is unexpectedly "
            "discarded.")]] inline constexpr auto
update<State, Output, UpdateTypes...>::k() const -> const gain & {
  return model.k;
}

template <typename State, typename Output, typename... UpdateTypes>
[[nodiscard("The returned innovation column vector Y is unexpectedly "
            "discarded.")]] inline constexpr auto
update<State, Output, UpdateTypes...>::y() const -> const innovation & {
  return model.y;
}

template <typename State, typename Output, typename... UpdateTypes>
[[nodiscard("The returned innovation uncertainty matrix S is unexpectedly "
            "discarded.")]] inline constexpr auto
update<State, Output, UpdateTypes...>::s() const
    -> const innovation_uncertainty & {
  return model.s;
}

template <typename State, typename Output, typename... UpdateTypes>
inline constexpr void
update<State, Output, UpdateTypes...>::observation(const auto &callable) {
  model.observation = callable;
}

template <typename State, typename Output, typename... UpdateTypes>
inline constexpr void
update<State, Output, UpdateTypes...>::operator()(const auto &...arguments) {
  model(arguments...);
}

template <typename State, typename Output, typename... UpdateTypes>
template <std::size_t Position>
[[nodiscard("The returned update argument is unexpectedly "
            "discarded.")]] inline constexpr auto
update<State, Output, UpdateTypes...>::operator()() const {
  return std::get<Position>(model.update_arguments);
}

} // namespace fcarouge

#endif // FCAROUGE_INTERNAL_UPDATE_TPP
