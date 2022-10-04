/*  __          _      __  __          _   _
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

#ifndef FCAROUGE_INTERNAL_KALMAN_HPP
#define FCAROUGE_INTERNAL_KALMAN_HPP

#include "utility.hpp"

#include <functional>
#include <ranges>
#include <tuple>
#include <type_traits>

namespace fcarouge::internal {

template <typename Identity> struct zero final {
  template <typename Type>
  [[nodiscard]] inline constexpr auto operator()() const -> Type {
    return 0 * Identity().template operator()<Type>();
  }

  template <std::ranges::range Type>
  [[nodiscard]] inline constexpr auto operator()() const -> Type {
    Type value;
    for (auto &&element : value) {
      element = 0;
    }
    return value;
  }

  // Specialize for arithmetic concept?
};

template <typename, typename, typename, typename, typename, typename, typename,
          typename>
struct kalman final {
  //! @todo Support some more specializations, all, or disable others?
};

template <typename State, typename Output, typename Transpose, typename Divide,
          typename Identity, typename... UpdateTypes,
          typename... PredictionTypes>
struct kalman<State, Output, void, Transpose, Divide, Identity,
              pack<UpdateTypes...>, pack<PredictionTypes...>> {
  template <typename Row, typename Column>
  using matrix = std::decay_t<std::invoke_result_t<Divide, Row, Column>>;
  using state = State;
  using output = Output;
  using input = empty;
  using estimate_uncertainty = matrix<state, state>;
  using process_uncertainty = matrix<state, state>;
  using output_uncertainty = matrix<output, output>;
  using state_transition = matrix<state, state>;
  using output_model = matrix<output, state>;
  using input_control = empty;
  using gain = matrix<state, output>;
  using innovation = output;
  using innovation_uncertainty = output_uncertainty;
  using observation_state_function =
      std::function<output_model(const state &, const UpdateTypes &...)>;
  using noise_observation_function = std::function<output_uncertainty(
      const state &, const output &, const UpdateTypes &...)>;
  using transition_state_function = std::function<state_transition(
      const state &, const PredictionTypes &...)>;
  using noise_process_function = std::function<process_uncertainty(
      const state &, const PredictionTypes &...)>;
  using transition_control_function = empty;
  using transition_function =
      std::function<state(const state &, const PredictionTypes &...)>;
  using observation_function =
      std::function<output(const state &, const UpdateTypes &...)>;
  using update_types = std::tuple<UpdateTypes...>;
  using prediction_types = std::tuple<PredictionTypes...>;

  //! @todo Is there a simpler way to initialize to the zero matrix?
  state x{zero<Identity>().template operator()<state>()};
  estimate_uncertainty p{
      Identity().template operator()<estimate_uncertainty>()};
  process_uncertainty q{
      zero<Identity>().template operator()<process_uncertainty>()};
  output_uncertainty r{
      zero<Identity>().template operator()<output_uncertainty>()};
  output_model h{Identity().template operator()<output_model>()};
  state_transition f{Identity().template operator()<state_transition>()};
  gain k{Identity().template operator()<gain>()};
  innovation y{zero<Identity>().template operator()<innovation>()};
  innovation_uncertainty s{
      Identity().template operator()<innovation_uncertainty>()};
  output z{zero<Identity>().template operator()<output>()};
  update_types update_arguments{};
  prediction_types prediction_arguments{};

  //! @todo Should we pass through the reference to the state x or have the user
  //! access it through filter.x() when needed? Where does the
  //! practical/performance tradeoff leans toward? For the general case? For the
  //! specialized cases? Same question applies to other parameters.
  //! @todo Pass the arguments by universal reference?
  observation_state_function observation_state_h{
      [&h = h](const state &state_x,
               const UpdateTypes &...update_pack) -> output_model {
        static_cast<void>(state_x);
        (static_cast<void>(update_pack), ...);
        return h;
      }};
  noise_observation_function noise_observation_r{
      [&r = r](const state &state_x, const output &output_z,
               const UpdateTypes &...update_pack) -> output_uncertainty {
        static_cast<void>(state_x);
        static_cast<void>(output_z);
        (static_cast<void>(update_pack), ...);
        return r;
      }};
  transition_state_function transition_state_f{
      [&f = f](const state &state_x,
               const PredictionTypes &...prediction_pack) -> state_transition {
        static_cast<void>(state_x);
        (static_cast<void>(prediction_pack), ...);
        return f;
      }};
  noise_process_function noise_process_q{
      [&q = q](const state &state_x, const PredictionTypes &...prediction_pack)
          -> process_uncertainty {
        static_cast<void>(state_x);
        (static_cast<void>(prediction_pack), ...);
        return q;
      }};
  transition_function transition{
      [&f = f](const state &state_x,
               const PredictionTypes &...prediction_pack) -> state {
        (static_cast<void>(prediction_pack), ...);
        return f * state_x;
      }};
  observation_function observation{
      [&h = h](const state &state_x,
               const UpdateTypes &...update_pack) -> output {
        (static_cast<void>(update_pack), ...);
        return h * state_x;
      }};

  Transpose transpose;
  Divide divide;
  Identity identity;

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

    const auto i{identity.template operator()<estimate_uncertainty>()};

    update_arguments = {update_pack...};
    z = output{output_z, outputs_z...};
    h = observation_state_h(x, update_pack...);
    r = noise_observation_r(x, z, update_pack...);
    s = innovation_uncertainty{h * p * transpose(h) + r};
    k = divide(p * transpose(h), s);
    y = z - observation(x, update_pack...);
    x = state{x + k * y};
    p = estimate_uncertainty{(i - k * h) * p * transpose(i - k * h) +
                             k * r * transpose(k)};
  }

  inline constexpr void predict(const PredictionTypes &...prediction_pack) {
    prediction_arguments = {prediction_pack...};
    f = transition_state_f(x, prediction_pack...);
    q = noise_process_q(x, prediction_pack...);
    x = transition(x, prediction_pack...);
    p = estimate_uncertainty{f * p * transpose(f) + q};
  }
};

template <typename State, typename Output, typename Input, typename Transpose,
          typename Divide, typename Identity, typename... UpdateTypes,
          typename... PredictionTypes>
struct kalman<State, Output, Input, Transpose, Divide, Identity,
              pack<UpdateTypes...>, pack<PredictionTypes...>> {
  template <typename Row, typename Column>
  using matrix = std::decay_t<std::invoke_result_t<Divide, Row, Column>>;
  using state = State;
  using output = Output;
  using input = Input;
  using estimate_uncertainty = matrix<state, state>;
  using process_uncertainty = matrix<state, state>;
  using output_uncertainty = matrix<output, output>;
  using state_transition = matrix<state, state>;
  using output_model = matrix<output, state>;
  using input_control = matrix<state, input>;
  using gain = matrix<state, output>;
  using innovation = output;
  using innovation_uncertainty = output_uncertainty;
  using observation_state_function =
      std::function<output_model(const state &, const UpdateTypes &...)>;
  using noise_observation_function = std::function<output_uncertainty(
      const state &, const output &, const UpdateTypes &...)>;
  using transition_state_function = std::function<state_transition(
      const state &, const input &, const PredictionTypes &...)>;
  using noise_process_function = std::function<process_uncertainty(
      const state &, const PredictionTypes &...)>;
  using transition_control_function =
      std::function<input_control(const PredictionTypes &...)>;
  using transition_function = std::function<state(const state &, const input &,
                                                  const PredictionTypes &...)>;
  using observation_function =
      std::function<output(const state &, const UpdateTypes &...)>;
  using update_types = std::tuple<UpdateTypes...>;
  using prediction_types = std::tuple<PredictionTypes...>;

  //! @todo Is there a simpler way to initialize to the zero matrix?
  state x{zero<Identity>().template operator()<state>()};
  estimate_uncertainty p{
      Identity().template operator()<estimate_uncertainty>()};
  process_uncertainty q{
      zero<Identity>().template operator()<process_uncertainty>()};
  output_uncertainty r{
      zero<Identity>().template operator()<output_uncertainty>()};
  output_model h{Identity().template operator()<output_model>()};
  state_transition f{Identity().template operator()<state_transition>()};
  input_control g{Identity().template operator()<input_control>()};
  gain k{Identity().template operator()<gain>()};
  innovation y{zero<Identity>().template operator()<innovation>()};
  innovation_uncertainty s{
      Identity().template operator()<innovation_uncertainty>()};
  output z{zero<Identity>().template operator()<output>()};
  input u{zero<Identity>().template operator()<input>()};
  update_types update_arguments{};
  prediction_types prediction_arguments{};

  //! @todo Should we pass through the reference to the state x or have the user
  //! access it through filter.x() when needed? Where does the
  //! practical/performance tradeoff leans toward? For the general case? For the
  //! specialized cases? Same question applies to other parameters.
  //! @todo Pass the arguments by universal reference?
  observation_state_function observation_state_h{
      [&h = h](const state &state_x,
               const UpdateTypes &...update_pack) -> output_model {
        static_cast<void>(state_x);
        (static_cast<void>(update_pack), ...);
        return h;
      }};
  noise_observation_function noise_observation_r{
      [&r = r](const state &state_x, const output &output_z,
               const UpdateTypes &...update_pack) -> output_uncertainty {
        static_cast<void>(state_x);
        static_cast<void>(output_z);
        (static_cast<void>(update_pack), ...);
        return r;
      }};
  transition_state_function transition_state_f{
      [&f = f](const state &state_x, const input &input_u,
               const PredictionTypes &...prediction_pack) -> state_transition {
        static_cast<void>(state_x);
        static_cast<void>(input_u);
        (static_cast<void>(prediction_pack), ...);
        return f;
      }};
  noise_process_function noise_process_q{
      [&q = q](const state &state_x, const PredictionTypes &...prediction_pack)
          -> process_uncertainty {
        static_cast<void>(state_x);
        (static_cast<void>(prediction_pack), ...);
        return q;
      }};
  transition_control_function transition_control_g{
      [&g = g](const PredictionTypes &...prediction_pack) -> input_control {
        (static_cast<void>(prediction_pack), ...);
        return g;
      }};
  transition_function transition{
      [&f = f, &g = g](const state &state_x, const input &input_u,
                       const PredictionTypes &...prediction_pack) -> state {
        (static_cast<void>(prediction_pack), ...);
        return f * state_x + g * input_u;
      }};
  observation_function observation{
      [&h = h](const state &state_x,
               const UpdateTypes &...update_pack) -> output {
        (static_cast<void>(update_pack), ...);
        return h * state_x;
      }};

  Transpose transpose;
  Divide divide;
  Identity identity;

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

    const auto i{identity.template operator()<estimate_uncertainty>()};

    update_arguments = {update_pack...};
    z = output{output_z, outputs_z...};
    h = observation_state_h(x, update_pack...);
    r = noise_observation_r(x, z, update_pack...);
    s = h * p * transpose(h) + r;
    k = divide(p * transpose(h), s);
    y = z - observation(x, update_pack...);
    x = state{x + k * y};
    p = estimate_uncertainty{(i - k * h) * p * transpose(i - k * h) +
                             k * r * transpose(k)};
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
    p = estimate_uncertainty{f * p * transpose(f) + q};
  }
};

} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_KALMAN_HPP
