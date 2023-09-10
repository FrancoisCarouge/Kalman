/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.3.0
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

#ifndef FCAROUGE_INTERNAL_KALMAN_HPP
#define FCAROUGE_INTERNAL_KALMAN_HPP

#include "function.hpp"
#include "utility.hpp"

#include <tuple>

namespace fcarouge::internal {
template <typename, typename, typename, typename, typename>
struct kalman final {
  //! @todo Support some more specializations, all, or disable others?
  //! @todo Add a pretty compilation error?
};

template <typename State, typename Output, typename... UpdateTypes,
          typename... PredictionTypes>
struct kalman<State, Output, void, pack<UpdateTypes...>,
              pack<PredictionTypes...>> {
  using state = State;
  using output = Output;
  using input = empty;
  using estimate_uncertainty = quotient<state, state>;
  using process_uncertainty = quotient<state, state>;
  using output_uncertainty = quotient<output, output>;
  using state_transition = quotient<state, state>;
  using output_model = quotient<output, state>;
  using input_control = empty;
  using gain = quotient<state, output>;
  using innovation = output;
  using innovation_uncertainty = output_uncertainty;
  using observation_state_function =
      function<output_model(const state &, const UpdateTypes &...)>;
  using noise_observation_function = function<output_uncertainty(
      const state &, const output &, const UpdateTypes &...)>;
  using transition_state_function =
      function<state_transition(const state &, const PredictionTypes &...)>;
  using noise_process_function =
      function<process_uncertainty(const state &, const PredictionTypes &...)>;
  using transition_control_function = empty;
  using transition_function =
      function<state(const state &, const PredictionTypes &...)>;
  using observation_function =
      function<output(const state &, const UpdateTypes &...)>;
  using update_types = std::tuple<UpdateTypes...>;
  using prediction_types = std::tuple<PredictionTypes...>;

  static inline const auto i{identity_v<quotient<state, state>>};

  state x{zero_v<state>};
  estimate_uncertainty p{identity_v<estimate_uncertainty>};
  process_uncertainty q{zero_v<process_uncertainty>};
  output_uncertainty r{zero_v<output_uncertainty>};
  output_model h{identity_v<output_model>};
  state_transition f{identity_v<state_transition>};
  gain k{identity_v<gain>};
  innovation y{zero_v<innovation>};
  innovation_uncertainty s{identity_v<innovation_uncertainty>};
  output z{zero_v<output>};
  update_types update_arguments{};
  prediction_types prediction_arguments{};
  transpose t{};

  //! @todo Should we pass through the reference to the state x or have the user
  //! access it through filter.x() when needed? Where does the
  //! practical/performance tradeoff leans toward? For the general case? For the
  //! specialized cases? Same question applies to other parameters.
  //! @todo Pass the arguments by universal reference?
  observation_state_function observation_state_h{
      [&h = h]([[maybe_unused]] const auto &...arguments) -> output_model {
        return h;
      }};
  noise_observation_function noise_observation_r{
      [&r =
           r]([[maybe_unused]] const auto &...arguments) -> output_uncertainty {
        return r;
      }};
  transition_state_function transition_state_f{
      [&f = f]([[maybe_unused]] const auto &...arguments) -> state_transition {
        return f;
      }};
  noise_process_function noise_process_q{
      [&q = q]([[maybe_unused]] const auto &...arguments)
          -> process_uncertainty { return q; }};
  transition_function transition{
      [&f = f](const state &state_x,
               [[maybe_unused]] const auto &...arguments) -> state {
        return f * state_x;
      }};
  observation_function observation{
      [&h = h](const state &state_x,
               [[maybe_unused]] const auto &...arguments) -> output {
        return h * state_x;
      }};

  //! @todo Do we want to store i - k * h in a temporary result for reuse? Or
  //! does the compiler/linker do it for us?
  //! @todo Do we want to support extended custom y = output_difference(z,
  //! observation(x))?
  //! @todo Do we want to pass z to `observation_state_h()`? What are the use
  //! cases?
  //! @todo Do we want to pass z to `observation()`? What are the use cases?
  //! @todo Use operator `+=` for the state update?
  template <typename Output0, typename... OutputN>
  inline constexpr void update(const UpdateTypes &...update_pack,
                               const Output0 &output_z,
                               const OutputN &...outputs_z) {
    update_arguments = {update_pack...};
    z = output{output_z, outputs_z...};
    h = observation_state_h(x, update_pack...);
    r = noise_observation_r(x, z, update_pack...);
    s = innovation_uncertainty{h * p * t(h) + r};
    k = p * t(h) / s;
    y = z - observation(x, update_pack...);
    x = state{x + k * y};
    p = estimate_uncertainty{(i - k * h) * p * t(i - k * h) + k * r * t(k)};
  }

  inline constexpr void predict(const PredictionTypes &...prediction_pack) {
    prediction_arguments = {prediction_pack...};
    f = transition_state_f(x, prediction_pack...);
    q = noise_process_q(x, prediction_pack...);
    x = transition(x, prediction_pack...);
    p = estimate_uncertainty{f * p * t(f) + q};
  }
};

template <typename State, typename Output, typename Input,
          typename... UpdateTypes, typename... PredictionTypes>
struct kalman<State, Output, Input, pack<UpdateTypes...>,
              pack<PredictionTypes...>> {
  using state = State;
  using output = Output;
  using input = Input;
  using estimate_uncertainty = quotient<state, state>;
  using process_uncertainty = quotient<state, state>;
  using output_uncertainty = quotient<output, output>;
  using state_transition = quotient<state, state>;
  using output_model = quotient<output, state>;
  using input_control = quotient<state, input>;
  using gain = quotient<state, output>;
  using innovation = output;
  using innovation_uncertainty = output_uncertainty;
  using observation_state_function =
      function<output_model(const state &, const UpdateTypes &...)>;
  using noise_observation_function = function<output_uncertainty(
      const state &, const output &, const UpdateTypes &...)>;
  using transition_state_function = function<state_transition(
      const state &, const input &, const PredictionTypes &...)>;
  using noise_process_function =
      function<process_uncertainty(const state &, const PredictionTypes &...)>;
  using transition_control_function =
      function<input_control(const PredictionTypes &...)>;
  using transition_function =
      function<state(const state &, const input &, const PredictionTypes &...)>;
  using observation_function =
      function<output(const state &, const UpdateTypes &...)>;
  using update_types = std::tuple<UpdateTypes...>;
  using prediction_types = std::tuple<PredictionTypes...>;

  static inline const auto i{identity_v<quotient<state, state>>};

  state x{zero_v<state>};
  estimate_uncertainty p{identity_v<estimate_uncertainty>};
  process_uncertainty q{zero_v<process_uncertainty>};
  output_uncertainty r{zero_v<output_uncertainty>};
  output_model h{identity_v<output_model>};
  state_transition f{identity_v<state_transition>};
  input_control g{identity_v<input_control>};
  gain k{identity_v<gain>};
  innovation y{zero_v<innovation>};
  innovation_uncertainty s{identity_v<innovation_uncertainty>};
  output z{zero_v<output>};
  input u{zero_v<input>};
  update_types update_arguments{};
  prediction_types prediction_arguments{};
  transpose t{};

  //! @todo Should we pass through the reference to the state x or have the user
  //! access it through filter.x() when needed? Where does the
  //! practical/performance tradeoff leans toward? For the general case? For the
  //! specialized cases? Same question applies to other parameters.
  //! @todo Pass the arguments by universal reference?
  observation_state_function observation_state_h{
      [&h = h]([[maybe_unused]] const auto &...arguments) -> output_model {
        return h;
      }};
  noise_observation_function noise_observation_r{
      [&r =
           r]([[maybe_unused]] const auto &...arguments) -> output_uncertainty {
        return r;
      }};
  transition_state_function transition_state_f{
      [&f = f]([[maybe_unused]] const auto &...arguments) -> state_transition {
        return f;
      }};
  noise_process_function noise_process_q{
      [&q = q]([[maybe_unused]] const auto &...arguments)
          -> process_uncertainty { return q; }};
  transition_control_function transition_control_g{
      [&g = g]([[maybe_unused]] const auto &...arguments) -> input_control {
        return g;
      }};
  transition_function transition{
      [&f = f, &g = g](const state &state_x, const input &input_u,
                       [[maybe_unused]] const auto &...arguments) -> state {
        return f * state_x + g * input_u;
      }};
  observation_function observation{
      [&h = h](const state &state_x,
               [[maybe_unused]] const auto &...arguments) -> output {
        return h * state_x;
      }};

  //! @todo Do we want to store i - k * h in a temporary result for reuse? Or
  //! does the compiler/linker do it for us?
  //! @todo Do we want to support extended custom y = output_difference(z,
  //! observation(x))?
  //! @todo Do we want to pass z to `observation_state_h()`? What are the use
  //! cases?
  //! @todo Do we want to pass z to `observation()`? What are the use cases?
  template <typename Output0, typename... OutputN>
  inline constexpr void update(const UpdateTypes &...update_pack,
                               const Output0 &output_z,
                               const OutputN &...outputs_z) {
    update_arguments = {update_pack...};
    z = output{output_z, outputs_z...};
    h = observation_state_h(x, update_pack...);
    r = noise_observation_r(x, z, update_pack...);
    s = h * p * t(h) + r;
    k = p * t(h) / s;
    y = z - observation(x, update_pack...);
    x = state{x + k * y};
    p = estimate_uncertainty{(i - k * h) * p * t(i - k * h) + k * r * t(k)};
  }

  //! @todo Extended support?
  //! @todo Should the transition state F computation arguments be  {x, u, args}
  //! instead of {x, args, u} or can we benefit for allowing passing through an
  //! input pack to the function? Similar parameter ordering question for
  //! related functions.
  //! @todo Do we want to pass u to `noise_process_q()`? What are the use cases?
  //! @todo Do we want to pass x, u to `transition_control_g()`? What are the
  //! use cases?
  template <typename Input0, typename... InputN>
  inline constexpr void predict(const PredictionTypes &...prediction_pack,
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
} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_KALMAN_HPP
