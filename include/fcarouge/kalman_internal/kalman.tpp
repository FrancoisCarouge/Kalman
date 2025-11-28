/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.3
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

#ifndef FCAROUGE_KALMAN_INTERNAL_KALMAN_TPP
#define FCAROUGE_KALMAN_INTERNAL_KALMAN_TPP

#include <tuple>
#include <type_traits>
#include <utility>

namespace fcarouge {
template <typename Filter>
template <typename... Arguments>
inline constexpr kalman<Filter>::kalman(Arguments... arguments)
    : filter{kalman_internal::filter<Filter>(arguments...)} {}

template <typename Filter>
inline constexpr decltype(auto) kalman<Filter>::x(this auto &&self,
                                                  const auto &...values)
  requires(kalman_internal::has_state<Filter>)
{
  if constexpr (sizeof...(values)) {
    self.filter.x = state{values...};
  }
  //! @todo A conditional no_discard woud be nice here.
  return std::forward<decltype(self)>(self).filter.x;
}

template <typename Filter>
inline constexpr decltype(auto) kalman<Filter>::z(this auto &&self,
                                                  const auto &...values)
  requires(kalman_internal::has_output<Filter>)
{
  if constexpr (sizeof...(values)) {
    self.filter.z = state{values...};
  }
  return std::forward<decltype(self)>(self).filter.z;
}

template <typename Filter>
inline constexpr decltype(auto) kalman<Filter>::u(this auto &&self,
                                                  const auto &...values)
  requires(kalman_internal::has_input<Filter>)
{
  if constexpr (sizeof...(values)) {
    self.filter.u = state{values...};
  }
  return std::forward<decltype(self)>(self).filter.u;
}

template <typename Filter>
inline constexpr decltype(auto) kalman<Filter>::p(this auto &&self,
                                                  const auto &...values)
  requires(kalman_internal::has_estimate_uncertainty<Filter>)
{
  if constexpr (sizeof...(values)) {
    self.filter.p = state{values...};
  }
  return std::forward<decltype(self)>(self).filter.p;
}

template <typename Filter>
inline constexpr decltype(auto) kalman<Filter>::q(this auto &&self,
                                                  const auto &...values)
  requires(kalman_internal::has_process_uncertainty<Filter>)
{
  if constexpr (sizeof...(values)) {
    if constexpr (std::is_convertible_v<decltype(values)...,
                                        typename Filter::process_uncertainty>) {
      self.filter.q = typename Filter::process_uncertainty{values...};
    } else {
      using noise_process_function = decltype(filter.noise_process_q);
      self.filter.noise_process_q = noise_process_function{values...};
    }
  }
  return std::forward<decltype(self)>(self).filter.q;
}

template <typename Filter>
inline constexpr decltype(auto) kalman<Filter>::r(this auto &&self,
                                                  const auto &...values)
  requires(kalman_internal::has_output_uncertainty<Filter>)
{
  if constexpr (sizeof...(values)) {
    if constexpr (std::is_convertible_v<decltype(values)...,
                                        typename Filter::output_uncertainty>) {
      self.filter.r = typename Filter::output_uncertainty{values...};
    } else {
      using noise_observation_function = decltype(filter.noise_observation_r);
      self.filter.noise_observation_r = noise_observation_function{values...};
    }
  }
  return std::forward<decltype(self)>(self).filter.r;
}

template <typename Filter>
inline constexpr decltype(auto) kalman<Filter>::f(this auto &&self,
                                                  const auto &...values)
  requires(kalman_internal::has_state_transition<Filter>)
{
  if constexpr (sizeof...(values)) {
    if constexpr (std::is_convertible_v<decltype(values)...,
                                        typename Filter::state_transition>) {
      self.filter.f = typename Filter::state_transition{values...};
    } else {
      using transition_state_function = decltype(filter.transition_state_f);
      self.filter.transition_state_f = transition_state_function{values...};
    }
  }
  return std::forward<decltype(self)>(self).filter.f;
}

template <typename Filter>
inline constexpr decltype(auto) kalman<Filter>::h(this auto &&self,
                                                  const auto &...values)
  requires(kalman_internal::has_output_model<Filter>)
{
  if constexpr (sizeof...(values)) {
    if constexpr (std::is_convertible_v<decltype(values)...,
                                        typename Filter::output_model>) {
      self.filter.h = typename Filter::output_model{values...};
    } else {
      using observation_state_function = decltype(filter.observation_state_h);
      self.filter.observation_state_h = observation_state_function{values...};
    }
  }
  return std::forward<decltype(self)>(self).filter.h;
}

template <typename Filter>
inline constexpr decltype(auto) kalman<Filter>::g(this auto &&self,
                                                  const auto &...values)
  requires(kalman_internal::has_input_control<Filter>)
{
  if constexpr (sizeof...(values)) {
    if constexpr (std::is_convertible_v<decltype(values)...,
                                        typename Filter::input_control>) {
      self.filter.g = typename Filter::input_control{values...};
    } else {
      using transition_control_function = decltype(filter.transition_control_g);
      self.filter.transition_control_g = transition_control_function{values...};
    }
  }
  return std::forward<decltype(self)>(self).filter.g;
}

template <typename Filter>
inline constexpr decltype(auto) kalman<Filter>::k(this auto &&self,
                                                  const auto &...values)
  requires(kalman_internal::has_gain<Filter>)
{
  if constexpr (sizeof...(values)) {
    self.filter.k = state{values...};
  }
  return std::forward<decltype(self)>(self).filter.k;
}

template <typename Filter>
inline constexpr decltype(auto) kalman<Filter>::y(this auto &&self,
                                                  const auto &...values)
  requires(kalman_internal::has_innovation<Filter>)
{
  if constexpr (sizeof...(values)) {
    self.filter.y = state{values...};
  }
  return std::forward<decltype(self)>(self).filter.y;
}

template <typename Filter>
inline constexpr decltype(auto) kalman<Filter>::s(this auto &&self,
                                                  const auto &...values)
  requires(kalman_internal::has_innovation_uncertainty<Filter>)
{
  if constexpr (sizeof...(values)) {
    self.filter.s = state{values...};
  }
  return std::forward<decltype(self)>(self).filter.s;
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

#endif // FCAROUGE_KALMAN_INTERNAL_KALMAN_TPP
