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

#ifndef FCAROUGE_INTERNAL_KALMAN_HPP
#define FCAROUGE_INTERNAL_KALMAN_HPP

#include "utility.hpp"

#include <functional>
#include <tuple>
#include <type_traits>

namespace fcarouge::internal {

template <typename, typename, typename, typename...> class update_model final {
  //! @todo Support some more specializations, all, or disable others?
};

template <typename State, typename Output, typename Divide,
          typename... UpdateTypes>
class update_model<State, Output, Divide, pack<UpdateTypes...>> final {
public:
  using state = State;
  using output = Output;
  using estimate_uncertainty = matrix<state, state>;
  using update_types = std::tuple<UpdateTypes...>;
  using output_model = matrix<output, state>;
  using output_uncertainty = matrix<output, output>;
  using innovation_uncertainty = output_uncertainty;
  using gain = matrix<state, output>;
  using innovation = output;
  using observation_state_function =
      std::function<output_model(const state &, const UpdateTypes &...)>;
  using noise_observation_function = std::function<output_uncertainty(
      const state &, const output &, const UpdateTypes &...)>;
  using observation_function =
      std::function<output(const state &, const UpdateTypes &...)>;

  static inline const auto i{identity_v<matrix<state, state>>};

  state x{zero_v<state>};
  estimate_uncertainty p{identity_v<estimate_uncertainty>};
  update_types update_arguments{};
  output z{zero_v<output>};
  output_model h{identity_v<output_model>};
  output_uncertainty r{zero_v<output_uncertainty>};
  innovation_uncertainty s{identity_v<innovation_uncertainty>};
  gain k{identity_v<gain>};
  innovation y{zero_v<innovation>};
  transpose t;
  Divide d;

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
  observation_function observation{
      [&h = h](const state &state_x,
               const UpdateTypes &...update_pack) -> output {
        (static_cast<void>(update_pack), ...);
        return h * state_x;
      }};

  template <typename Output0, typename... OutputN>
  inline constexpr void operator()(const UpdateTypes &...update_pack,
                                   const Output0 &output_z,
                                   const OutputN &...outputs_z) {
    update_arguments = {update_pack...};
    z = output{output_z, outputs_z...};
    h = observation_state_h(x, update_pack...);
    r = noise_observation_r(x, z, update_pack...);
    s = innovation_uncertainty{h * p * t(h) + r};
    k = divide(p * t(h), s);
    y = z - observation(x, update_pack...);
    x = state{x + k * y};
    p = estimate_uncertainty{(i - k * h) * p * t(i - k * h) + k * r * t(k)};
  }

  template <std::size_t Position> inline constexpr auto operator()() const {
    return std::get<Position>(update_arguments);
  }
};

template <typename, typename, typename...> class prediction_model final {
  //! @todo Support some more specializations, all, or disable others?
};

template <typename State, typename Input, typename... PredictionTypes>
class prediction_model<State, Input, pack<PredictionTypes...>> final {
public:
  using state = State;
  using input = Input;
  using estimate_uncertainty = matrix<state, state>;
  using process_uncertainty = matrix<state, state>;
  using state_transition = matrix<state, state>;
  using prediction_types = std::tuple<PredictionTypes...>;
  using input_control = matrix<state, input>;
  using transition_state_function = std::function<state_transition(
      const state &, const input &, const PredictionTypes &...)>;
  using noise_process_function = std::function<process_uncertainty(
      const state &, const PredictionTypes &...)>;
  using transition_control_function =
      std::function<input_control(const PredictionTypes &...)>;
  using transition_function = std::function<state(const state &, const input &,
                                                  const PredictionTypes &...)>;

  state x{zero_v<state>};
  estimate_uncertainty p{identity_v<estimate_uncertainty>};
  prediction_types prediction_arguments{};
  process_uncertainty q{zero_v<process_uncertainty>};
  state_transition f{identity_v<state_transition>};
  input_control g{identity_v<input_control>};
  input u{zero_v<input>};
  transpose t;
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

  template <std::size_t Position> inline constexpr auto operator()() const {
    return std::get<Position>(prediction_arguments);
  }
};

template <typename State, typename... PredictionTypes>
class prediction_model<State, void, pack<PredictionTypes...>> final {
public:
  using state = State;
  using input = empty;
  using estimate_uncertainty = matrix<state, state>;
  using process_uncertainty = matrix<state, state>;
  using state_transition = matrix<state, state>;
  using prediction_types = std::tuple<PredictionTypes...>;
  using input_control = empty;
  using transition_state_function = std::function<state_transition(
      const state &, const PredictionTypes &...)>;
  using noise_process_function = std::function<process_uncertainty(
      const state &, const PredictionTypes &...)>;
  using transition_control_function = empty;
  using transition_function =
      std::function<state(const state &, const PredictionTypes &...)>;

  state x{zero_v<state>};
  estimate_uncertainty p{identity_v<estimate_uncertainty>};
  prediction_types prediction_arguments{};
  process_uncertainty q{zero_v<process_uncertainty>};
  state_transition f{identity_v<state_transition>};
  transpose t;
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

  inline constexpr auto operator()(const PredictionTypes &...prediction_pack) {
    prediction_arguments = {prediction_pack...};
    f = transition_state_f(x, prediction_pack...);
    q = noise_process_q(x, prediction_pack...);
    x = transition(x, prediction_pack...);
    p = estimate_uncertainty{f * p * t(f) + q};
  }

  template <std::size_t Position> inline constexpr auto operator()() const {
    return std::get<Position>(prediction_arguments);
  }
};

template <typename UpdateModel, typename PredictionModel> struct kalman {
  // ASSERT STATE/UNCERTAINTY TYPE COMPATIBILITY?
  using state = typename UpdateModel::state;
  using output = typename UpdateModel::output;
  using input = typename PredictionModel::input;
  using estimate_uncertainty = typename UpdateModel::estimate_uncertainty;
  using process_uncertainty = typename PredictionModel::process_uncertainty;
  using output_uncertainty = typename UpdateModel::output_uncertainty;
  using state_transition = typename PredictionModel::state_transition;
  using output_model = typename UpdateModel::output_model;
  using input_control = typename PredictionModel::input_control;
  using gain = typename UpdateModel::gain;
  using innovation = typename UpdateModel::innovation;
  using innovation_uncertainty = typename UpdateModel::innovation_uncertainty;

  state xx{zero_v<state>}; // REMOVE! //////////////////////////////////////////
  estimate_uncertainty pp{identity_v<estimate_uncertainty>}; // REMOVE! ////////

  UpdateModel updator{};
  PredictionModel predictor{};

  inline constexpr auto x() const -> const state & { return xx; }

  inline constexpr auto x() -> state & { return xx; }

  inline constexpr void x(const auto &value, const auto &...values) {
    xx = {value, values...};
  }

  inline constexpr auto z() const -> const output & { return updator.z(); }

  inline constexpr auto u() const
      -> const input &requires(not std::is_same_v<input, empty>) {
                        return predictor.u();
                      }

  inline constexpr auto p() const -> const estimate_uncertainty & {
    return pp;
  }

  inline constexpr auto p() -> estimate_uncertainty & { return pp; }

  inline constexpr void p(const auto &value, const auto &...values) {
    pp = {value, values...};
  }

  inline constexpr auto q() const -> const process_uncertainty & {
    return predictor.q();
  }

  inline constexpr auto q() -> process_uncertainty & { return predictor.q(); }

  inline constexpr void q(const auto &value, const auto &...values) {
    predictor.q(value, values...);
  }

  inline constexpr auto r() const -> const output_uncertainty & {
    return updator.r();
  }

  inline constexpr auto r() -> output_uncertainty & { return updator.r(); }

  inline constexpr void r(const auto &value, const auto &...values) {
    updator.r(value, values...);
  }

  inline constexpr auto f() const -> const state_transition & {
    return predictor.f();
  }

  inline constexpr auto f() -> state_transition & { return predictor.f(); }

  inline constexpr void f(const auto &value, const auto &...values) {
    predictor.f(value, values...);
  }

  inline constexpr auto h() const -> const output_model & {
    return updator.h();
  }

  inline constexpr auto h() -> output_model & { return updator.h(); }

  inline constexpr void h(const auto &value, const auto &...values) {
    updator.h(value, values...);
  }

  inline constexpr auto g() const -> const input_control &requires(
      not std::is_same_v<input_control, empty>) { return predictor.g(); }

  inline constexpr auto g()
      -> input_control &requires(not std::is_same_v<input_control, empty>) {
                          return predictor.g();
                        }

  inline constexpr void g(const auto &value, const auto &...values)
    requires(not std::is_same_v<input_control, empty>)
  {
    predictor.g(value, values...);
  }

  inline constexpr auto k() const -> const gain & { return updator.k(); }

  inline constexpr auto y() const -> const innovation & { return updator.y(); }

  inline constexpr auto s() const -> const innovation_uncertainty & {
    return updator.s();
  }

  inline constexpr void transition(auto &&callable) {
    predictor.transition(std::forward<decltype(callable)>(callable));
  }

  inline constexpr void observation(auto &&callable) {
    updator.observation(std::forward<decltype(callable)>(callable));
  }

  //! @todo Extended support?
  //! @todo Should the transition state F computation arguments be  {x, u, args}
  //! instead of {x, args, u} or can we benefit for allowing passing through an
  //! input pack to the function? Similar parameter ordering question for
  //! related functions.
  //! @todo Do we want to pass u to `noise_process_q()`? What are the use cases?
  //! @todo Do we want to pass x, u to `transition_control_g()`? What are the
  //! use cases?
  inline constexpr void predict(const auto &...arguments) {
    predictor(arguments...);
  }

  template <std::size_t Position> inline constexpr auto predict() const {
    return predictor.template operator()<Position>();
  }

  //! @todo Do we want to store i - k * h in a temporary result for reuse? Or
  //! does the compiler/linker do it for us?
  //! @todo Do we want to support extended custom y = output_difference(z,
  //! observation(x))?
  //! @todo Do we want to pass z to `observation_state_h()`? What are the use
  //! cases?
  //! @todo Do we want to pass z to `observation()`? What are the use cases?
  inline constexpr void update(const auto &...arguments) {
    updator(arguments...);
  }

  template <std::size_t Position> inline constexpr auto update() const {
    return updator.template operator()<Position>();
  }
};

} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_KALMAN_HPP
