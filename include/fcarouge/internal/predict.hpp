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

#ifndef FCAROUGE_INTERNAL_PREDICT_HPP
#define FCAROUGE_INTERNAL_PREDICT_HPP

#include "utility.hpp"

#include <functional>
#include <tuple>

namespace fcarouge {
namespace internal {

template <typename State, typename Input, typename... PredictionTypes>
struct predict final {
  //! @todo Add a pretty compilation error?
};

template <typename State, typename... PredictionTypes>
struct predict<State, void, PredictionTypes...> {
  using state = State;
  using input = empty;
  using estimate_uncertainty = matrix<state, state>;
  using process_uncertainty = matrix<state, state>;
  using state_transition = matrix<state, state>;
  using input_control = empty;
  using transition_state_function = std::function<state_transition(
      const state &, const input &, const PredictionTypes &...)>;
  using noise_process_function = std::function<process_uncertainty(
      const state &, const PredictionTypes &...)>;
  using transition_control_function = empty;
  using transition_function = std::function<state(const state &, const input &,
                                                  const PredictionTypes &...)>;
  using prediction_types = std::tuple<PredictionTypes...>;

  state x{zero_v<state>};
  estimate_uncertainty p{identity_v<estimate_uncertainty>};
  process_uncertainty q{zero_v<process_uncertainty>};
  prediction_types prediction_arguments{};
  transpose t{};

  transition_state_function transition_state_f{
      [&f = f]([[maybe_unused]] const state &state_x,
               [[maybe_unused]] const PredictionTypes &...prediction_pack)
          -> state_transition { return f; }};
  noise_process_function noise_process_q{
      [&q = q]([[maybe_unused]] const state &state_x,
               [[maybe_unused]] const PredictionTypes &...prediction_pack)
          -> process_uncertainty { return q; }};
  transition_function transition{
      [&f = f](const state &state_x,
               [[maybe_unused]] const PredictionTypes &...prediction_pack)
          -> state { return f * state_x; }};

  inline constexpr void operator()(const PredictionTypes &...prediction_pack) {
    prediction_arguments = {prediction_pack...};
    f = transition_state_f(x, prediction_pack...);
    q = noise_process_q(x, prediction_pack...);
    x = transition(x, prediction_pack...);
    p = estimate_uncertainty{f * p * t(f) + q};
  }
};

template <typename State, typename Input, typename... PredictionTypes>
struct predict<State, Input, PredictionTypes...> {
  using state = State;
  using input = Input;
  using estimate_uncertainty = matrix<state, state>;
  using process_uncertainty = matrix<state, state>;
  using state_transition = matrix<state, state>;
  using input_control = matrix<state, input>;
  using transition_state_function = std::function<state_transition(
      const state &, const input &, const PredictionTypes &...)>;
  using noise_process_function = std::function<process_uncertainty(
      const state &, const PredictionTypes &...)>;
  using transition_control_function =
      std::function<input_control(const PredictionTypes &...)>;
  using transition_function = std::function<state(const state &, const input &,
                                                  const PredictionTypes &...)>;
  using prediction_types = std::tuple<PredictionTypes...>;

  state x{zero_v<state>};
  estimate_uncertainty p{identity_v<estimate_uncertainty>};
  process_uncertainty q{zero_v<process_uncertainty>};
  state_transition f{identity_v<state_transition>};
  input_control g{identity_v<input_control>};
  input u{zero_v<input>};
  prediction_types prediction_arguments{};
  transpose t{};

  transition_state_function transition_state_f{
      [&f = f]([[maybe_unused]] const state &state_x,
               [[maybe_unused]] const input &input_u,
               [[maybe_unused]] const PredictionTypes &...prediction_pack)
          -> state_transition { return f; }};
  noise_process_function noise_process_q{
      [&q = q]([[maybe_unused]] const state &state_x,
               [[maybe_unused]] const PredictionTypes &...prediction_pack)
          -> process_uncertainty { return q; }};
  transition_control_function transition_control_g{
      [&g = g]([[maybe_unused]] const PredictionTypes &...prediction_pack)
          -> input_control { return g; }};
  transition_function transition{
      [&f = f, &g = g](
          const state &state_x, const input &input_u,
          [[maybe_unused]] const PredictionTypes &...prediction_pack) -> state {
        return f * state_x + g * input_u;
      }};

  template <typename Input0, typename... InputN>
  inline constexpr void operator()(const PredictionTypes &...prediction_pack,
                                   const Input0 &input_u,
                                   const InputN &...inputs_u) {

    prediction_arguments = {prediction_pack...};
    u = input{input_u, inputs_u...};
    f = transition_state_f(x, u, prediction_pack...);
    q = noise_process_q(x, prediction_pack...);
    g = transition_control_g(prediction_pack...);
    x = transition(x, u, prediction_pack...);
    p = estimate_uncertainty{f * p * t(f) + q};
  }
};

} // namespace internal
} // namespace fcarouge

#endif // FCAROUGE_INTERNAL_PREDICT_HPP
