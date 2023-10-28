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
#include <type_traits>

namespace fcarouge::internal {
// Helper template to support multiple pack deduction.
template <typename, typename, typename, typename>
struct kalman_s_o_us_ps final {};

template <typename State, typename Output, typename... UpdateTypes,
          typename... PredictionTypes>
struct kalman_s_o_us_ps<State, Output, pack<UpdateTypes...>,
                        pack<PredictionTypes...>> {
  using state = State;
  using output = Output;
  using estimate_uncertainty = quotient<state, state>;
  using process_uncertainty = quotient<state, state>;
  using output_uncertainty = quotient<output, output>;
  using state_transition = quotient<state, state>;
  using output_model = quotient<output, state>;
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
      [&hh = h]([[maybe_unused]] const auto &...arguments) -> output_model {
        return hh;
      }};
  noise_observation_function noise_observation_r{
      [&rr =
           r]([[maybe_unused]] const auto &...arguments) -> output_uncertainty {
        return rr;
      }};
  transition_state_function transition_state_f{
      [&ff = f]([[maybe_unused]] const auto &...arguments) -> state_transition {
        return ff;
      }};
  noise_process_function noise_process_q{
      [&qq = q]([[maybe_unused]] const auto &...arguments)
          -> process_uncertainty { return qq; }};
  transition_function transition{
      [&ff = f](const state &state_x,
                [[maybe_unused]] const auto &...arguments) -> state {
        return ff * state_x;
      }};
  observation_function observation{
      [&hh = h](const state &state_x,
                [[maybe_unused]] const auto &...arguments) -> output {
        return hh * state_x;
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

// Helper template to support multiple pack deduction.
template <typename, typename, typename, typename, typename>
struct kalman_s_o_i_us_ps final {};

template <typename State, typename Output, typename Input,
          typename... UpdateTypes, typename... PredictionTypes>
struct kalman_s_o_i_us_ps<State, Output, Input, pack<UpdateTypes...>,
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
  input u{zero_v<input>};
  output_model h{identity_v<output_model>};
  state_transition f{identity_v<state_transition>};
  input_control g{identity_v<input_control>};
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
      [&hh = h]([[maybe_unused]] const auto &...arguments) -> output_model {
        return hh;
      }};
  noise_observation_function noise_observation_r{
      [&rr =
           r]([[maybe_unused]] const auto &...arguments) -> output_uncertainty {
        return rr;
      }};
  transition_state_function transition_state_f{
      [&ff = f]([[maybe_unused]] const auto &...arguments) -> state_transition {
        return ff;
      }};
  noise_process_function noise_process_q{
      [&qq = q]([[maybe_unused]] const auto &...arguments)
          -> process_uncertainty { return qq; }};
  transition_control_function transition_control_g{
      [&gg = g]([[maybe_unused]] const auto &...arguments) -> input_control {
        return gg;
      }};
  transition_function transition{
      [&ff = f, &gg = g](const state &state_x, const input &input_u,
                         [[maybe_unused]] const auto &...arguments) -> state {
        return ff * state_x + gg * input_u;
      }};
  observation_function observation{
      [&hh = h](const state &state_x,
                [[maybe_unused]] const auto &...arguments) -> output {
        return hh * state_x;
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
//! @todo What is the minimal meaningful filter? Should further details be
//! optimized out?
template <typename State, typename Output> struct kalman_s_o {
  using state = State;
  using output = Output;
  using estimate_uncertainty = quotient<state, state>;
  using process_uncertainty = quotient<state, state>;
  using output_uncertainty = quotient<output, output>;
  using state_transition = quotient<state, state>;
  using gain = quotient<state, output>;
  using innovation = output;
  using innovation_uncertainty = output_uncertainty;
  using noise_observation_function =
      function<output_uncertainty(const state &, const output &)>;
  using transition_state_function = function<state_transition(const state &)>;
  using noise_process_function = function<process_uncertainty(const state &)>;
  using transition_function = function<state(const state &)>;
  using observation_function = function<output(const state &)>;

  static inline const auto i{identity_v<quotient<state, state>>};

  state x{zero_v<state>};
  estimate_uncertainty p{identity_v<estimate_uncertainty>};
  output_uncertainty r{zero_v<output_uncertainty>};

  process_uncertainty q{zero_v<process_uncertainty>}; ////// REMOVE ME
  state_transition f{identity_v<state_transition>};
  gain k{identity_v<gain>};
  innovation y{zero_v<innovation>};
  innovation_uncertainty s{identity_v<innovation_uncertainty>};
  output z{zero_v<output>};
  transpose t{};

  noise_observation_function noise_observation_r{
      [&rr =
           r]([[maybe_unused]] const auto &...arguments) -> output_uncertainty {
        return rr;
      }};
  transition_state_function transition_state_f{
      [&ff = f]([[maybe_unused]] const auto &...arguments) -> state_transition {
        return ff;
      }};
  noise_process_function noise_process_q{
      [&qq = q]([[maybe_unused]] const auto &...arguments)
          -> process_uncertainty { return qq; }};
  transition_function transition{
      [&ff = f](const state &state_x,
                [[maybe_unused]] const auto &...arguments) -> state {
        return ff * state_x;
      }};

  inline constexpr void update(const output &output_z) {
    z = output_z;
    r = noise_observation_r(x, z);
    s = innovation_uncertainty{p + r};
    k = p / s;
    y = z - x;
    x = state{x + k * y};
    p = estimate_uncertainty{(i - k) * p * t(i - k) + k * r * t(k)};
  }

  inline constexpr void predict() {
    f = transition_state_f(x);
    q = noise_process_q(x);
    x = transition(x);
    p = estimate_uncertainty{f * p * t(f) + q};
  }
};

template <typename Type> struct state {
  Type value;

  // Remove if/not array?
  explicit state(Type v) : value{v} {}

  // same?
  template <typename... Types>
    requires(sizeof...(Types) > 1)
  state(Types... elements) : value{elements...} {}
};

template <typename Type> state(Type) -> state<Type>;

template <typename... Types>
state(Types... elements)
    -> state<std::remove_cvref_t<first_t<Types...>>[sizeof...(Types)]>;

template <typename Type> struct estimate_uncertainty {
  Type value;

  // Remove if/not array?
  explicit estimate_uncertainty(Type v) : value{v} {}

  template <typename... Types, auto... Columns>
  estimate_uncertainty([[maybe_unused]] const Types (&...rows)[Columns])
  // requires(std::conjunction_v<std::is_same<first_t<Types...>, Types>...>
  // &&
  //          ((Columns == first_v<Columns>) && ... && true))
  {
    int i{0};
    (
        [&]([[maybe_unused]] auto row) {
          // for constexpr?
          for (std::remove_cv_t<decltype(first_v<Columns...>)> j{0};
               j < first_v<Columns...>; ++j) {
            value[i * first_v<Columns...> + j] = row[j];
          }
          ++i;
        }(rows),
        ...);
  }
};

template <typename Type>
estimate_uncertainty(Type) -> estimate_uncertainty<Type>;

template <typename... Types, auto... Columns>
estimate_uncertainty([[maybe_unused]] const Types (&...rows)[Columns])
    -> estimate_uncertainty<std::remove_cvref_t<
        first_t<Types...>>[first_v<Columns...> * sizeof...(rows)]>;

template <typename Type> struct output_uncertainty {
  Type value;

  explicit output_uncertainty(Type v) : value{v} {}

  template <typename... Types, auto... Columns>
  output_uncertainty([[maybe_unused]] const Types (&...rows)[Columns]) {
    int i{0};
    (
        [&]([[maybe_unused]] auto row) {
          // for constexpr?
          for (std::remove_cv_t<decltype(first_v<Columns...>)> j{0};
               j < first_v<Columns...>; ++j) {
            value[i * first_v<Columns...> + j] = row[j];
          }
          ++i;
        }(rows),
        ...);
  }
};

template <typename Type> output_uncertainty(Type) -> output_uncertainty<Type>;

template <typename... Types, auto... Columns>
output_uncertainty([[maybe_unused]] const Types (&...rows)[Columns])
    -> output_uncertainty<std::remove_cvref_t<
        first_t<Types...>>[first_v<Columns...> * sizeof...(rows)
                           // first_v<Columns...>
]>;

template <typename Type> struct process_uncertainty {
  Type value;

  // Remove these if type not an array? Fall back to simpler structure?
  explicit process_uncertainty(Type v) : value{v} {}

  template <typename... Types, auto... Columns>
  process_uncertainty([[maybe_unused]] const Types (&...rows)[Columns]) {
    int i{0};
    (
        // for constexpr?
        [&]([[maybe_unused]] auto row) {
          for (std::remove_cv_t<decltype(first_v<Columns...>)> j{0};
               j < first_v<Columns...>; ++j) {
            value[i * first_v<Columns...> + j] = row[j];
          }
          ++i;
        }(rows),
        ...);
  }
};

template <typename Type> process_uncertainty(Type) -> process_uncertainty<Type>;

template <typename... Types, auto... Columns>
process_uncertainty([[maybe_unused]] const Types (&...rows)[Columns])
    -> process_uncertainty<std::remove_cvref_t<
        first_t<Types...>>[first_v<Columns...> * sizeof...(rows)]>;

template <typename Type> struct input_t {};

template <typename Type> inline input_t<Type> input{};

template <typename Type> struct output_t {};

template <typename Type> inline output_t<Type> output{};

template <typename Type> struct output_model {
  Type value;

  explicit output_model(Type v) : value{v} {}

  template <typename... Types, auto... Columns>
  output_model([[maybe_unused]] const Types (&...rows)[Columns]) {
    int i{0};
    (
        [&]([[maybe_unused]] auto row) {
          for (std::remove_cv_t<decltype(first_v<Columns...>)> j{0};
               j < first_v<Columns...>; ++j) {
            value[i * first_v<Columns...> + j] = row[j];
          }
          ++i;
        }(rows),
        ...);
  }
};

template <typename Type> output_model(Type) -> output_model<Type>;

template <typename... Types, auto... Columns>
output_model([[maybe_unused]] const Types (&...rows)[Columns])
    -> output_model<std::remove_cvref_t<first_t<Types...>>[first_v<Columns...> *
                                                           sizeof...(rows)]>;

template <typename Type> struct state_transition {
  Type value;

  explicit state_transition(Type v) : value{v} {}

  template <typename... Types, auto... Columns>
  state_transition([[maybe_unused]] const Types (&...rows)[Columns]) {
    int i{0};
    (
        [&]([[maybe_unused]] auto row) {
          for (std::remove_cv_t<decltype(first_v<Columns...>)> j{0};
               j < first_v<Columns...>; ++j) {
            value[i * first_v<Columns...> + j] = row[j];
          }
          ++i;
        }(rows),
        ...);
  }
};

template <typename Type> state_transition(Type) -> state_transition<Type>;

template <typename... Types, auto... Columns>
state_transition([[maybe_unused]] const Types (&...rows)[Columns])
    -> state_transition<std::remove_cvref_t<
        first_t<Types...>>[first_v<Columns...> * sizeof...(rows)]>;

//! @todo Better name not ending by *_types?
template <typename... Types> struct update_types_t {};

template <typename... Types> inline update_types_t<Types...> update_types{};

template <typename... Types> struct prediction_types_t {};

template <typename... Types>
inline prediction_types_t<Types...> prediction_types{};

//! @todo Support arbritary order of configuration parameters?
//! @todo Support user defined types by type name reflection? Case, naming
//! convention insensitive?
//! @todo Some of these overload should probably be removed to guide the user
//! towards better initialization practices?
template <typename Filter = void> struct make_filter {
  template <typename... Arguments>
  inline constexpr auto
  operator()([[maybe_unused]] Arguments... arguments) const
    requires(std::same_as<Filter, void>)
  {
    static_assert(false,
                  "This requested filter configuration is not yet supported. "
                  "Please, submit a pull request or feature request.");
    // TODO Dumb out dummy: nil filter.
    return kalman_s_o<double, double>{};
  }

  //! @todo Rename, clean the concet: undeduced?
  inline constexpr auto operator()() const -> kalman_s_o<double, double>
    requires(std::same_as<Filter, void>);

  inline constexpr auto operator()() const { return Filter{}; }

  template <typename State, typename Input>
  inline constexpr auto operator()(state<State> x,
                                   [[maybe_unused]] input_t<Input> u) const {
    return kalman_s_o_i_us_ps<State, State, Input, empty_pack, empty_pack>{
        x.value};
  }

  template <typename State, typename Output>
  inline constexpr auto operator()(state<State> x,
                                   [[maybe_unused]] output_t<Output> z) const {
    return kalman_s_o_us_ps<State, Output, empty_pack, empty_pack>{x.value};
  }

  template <typename State, typename Output, typename... Us, typename... Ps>
  inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] output_t<Output> z,
             [[maybe_unused]] update_types_t<Us...> uts,
             [[maybe_unused]] prediction_types_t<Ps...> pts) const {

    using kt = kalman_s_o_us_ps<State, Output, repack_t<update_types_t<Us...>>,
                                repack_t<prediction_types_t<Ps...>>>;
    return kt{typename kt::state{x.value}};
  }

  template <typename State, typename Output, typename Input>
  inline constexpr auto operator()(state<State> x,
                                   [[maybe_unused]] output_t<Output> z,
                                   [[maybe_unused]] input_t<Input> u) const {
    using kt = kalman_s_o_i_us_ps<State, Output, Input, empty_pack, empty_pack>;
    return kt{typename kt::state{x.value}};
  }

  template <typename State, typename Output, typename Input,
            typename EstimateUncertainty, typename... Ps>
  inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] output_t<Output> z,
             [[maybe_unused]] input_t<Input> u,
             estimate_uncertainty<EstimateUncertainty> p,
             [[maybe_unused]] prediction_types_t<Ps...> pts) const {
    using kt = kalman_s_o_i_us_ps<State, Output, Input, empty_pack,
                                  repack_t<prediction_types_t<Ps...>>>;
    return kt{typename kt::state{x.value},
              typename kt::estimate_uncertainty{p.value}};
  }

  template <typename State, typename Output, typename Input, typename... Us,
            typename... Ps>
  inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] output_t<Output> z,
             [[maybe_unused]] input_t<Input> u,
             [[maybe_unused]] update_types_t<Us...> uts,
             [[maybe_unused]] prediction_types_t<Ps...> pts) const {
    using kt = kalman_s_o_i_us_ps<State, Output, Input,
                                  repack_t<update_types_t<Us...>>,
                                  repack_t<prediction_types_t<Ps...>>>;
    return kt{typename kt::state{x.value}};
  }

  template <typename State, typename Output, typename EstimateUncertainty,
            typename... Us, typename... Ps>
  inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] output_t<Output> z,
             estimate_uncertainty<EstimateUncertainty> p,
             [[maybe_unused]] update_types_t<Us...> uts,
             [[maybe_unused]] prediction_types_t<Ps...> pts) const {
    using kt = kalman_s_o_us_ps<State, Output, repack_t<update_types_t<Us...>>,
                                repack_t<prediction_types_t<Ps...>>>;
    return kt{typename kt::state{x.value},
              typename kt::estimate_uncertainty{p.value}};
  }

  template <typename State, typename Output, typename EstimateUncertainty,
            typename ProcessUncertainty, typename... Us, typename... Ps>
  inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] output_t<Output> z,
             estimate_uncertainty<EstimateUncertainty> p,
             process_uncertainty<ProcessUncertainty> q) const {
    using kt = kalman_s_o_us_ps<State, Output, empty_pack, empty_pack>;
    return kt{typename kt::state{x.value},
              typename kt::estimate_uncertainty{p.value},
              typename kt::process_uncertainty{q.value}};
  }

  template <typename State, typename Output, typename EstimateUncertainty,
            typename ProcessUncertainty, typename... Us, typename... Ps>
  inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] output_t<Output> z,
             estimate_uncertainty<EstimateUncertainty> p,
             process_uncertainty<ProcessUncertainty> q,
             [[maybe_unused]] update_types_t<Us...> uts,
             [[maybe_unused]] prediction_types_t<Ps...> pts) const {
    using kt = kalman_s_o_us_ps<State, Output, repack_t<update_types_t<Us...>>,
                                repack_t<prediction_types_t<Ps...>>>;
    return kt{typename kt::state{x.value},
              typename kt::estimate_uncertainty{p.value},
              typename kt::process_uncertainty{q.value}};
  }

  template <typename State, typename Output, typename EstimateUncertainty,
            typename ProcessUncertainty, typename OutputUncertainty,
            typename... Us, typename... Ps>
  inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] output_t<Output> z,
             estimate_uncertainty<EstimateUncertainty> p,
             process_uncertainty<ProcessUncertainty> q,
             output_uncertainty<OutputUncertainty> r,
             [[maybe_unused]] update_types_t<Us...> uts,
             [[maybe_unused]] prediction_types_t<Ps...> pts) const {
    using kt = kalman_s_o_us_ps<State, Output, repack_t<update_types_t<Us...>>,
                                repack_t<prediction_types_t<Ps...>>>;
    return kt{typename kt::state{x.value},
              typename kt::estimate_uncertainty{p.value},
              typename kt::process_uncertainty{q.value},
              typename kt::output_uncertainty{r.value}};
  }

  template <typename State, typename Output, typename EstimateUncertainty,
            typename ProcessUncertainty, typename OutputUncertainty,
            typename OutputModel, typename StateTransition>
  inline constexpr auto operator()(state<State> x,
                                   [[maybe_unused]] output_t<Output> z,
                                   estimate_uncertainty<EstimateUncertainty> p,
                                   process_uncertainty<ProcessUncertainty> q,
                                   output_uncertainty<OutputUncertainty> r,
                                   output_model<OutputModel> h,
                                   state_transition<StateTransition> f) const {
    using kt = kalman_s_o_us_ps<State, Output, empty_pack, empty_pack>;
    return kt{
        typename kt::state{x.value},
        typename kt::estimate_uncertainty{p.value},
        typename kt::process_uncertainty{q.value},
        typename kt::output_uncertainty{r.value},
        typename kt::output_model{h.value},
        typename kt::state_transition{f.value},
    };
  }

  template <typename State, typename EstimateUncertainty,
            typename OutputUncertainty>
  inline constexpr auto
  operator()(state<State> x, estimate_uncertainty<EstimateUncertainty> p,
             output_uncertainty<OutputUncertainty> r) const {
    using kt = kalman_s_o<State, State>;
    return kt{typename kt::state{x.value},
              typename kt::estimate_uncertainty{p.value},
              typename kt::output_uncertainty{r.value}};
  }

  template <typename State, typename EstimateUncertainty,
            typename OutputUncertainty, typename ProcessUncertainty>
  inline constexpr auto
  operator()(state<State> x, estimate_uncertainty<EstimateUncertainty> p,
             process_uncertainty<ProcessUncertainty> q,
             output_uncertainty<OutputUncertainty> r) const {
    using kt = kalman_s_o_us_ps<State, State, empty_pack, empty_pack>;
    return kt{typename kt::state{x.value},
              typename kt::estimate_uncertainty{p.value},
              typename kt::process_uncertainty{q.value},
              typename kt::output_uncertainty{r.value}};
  }

  template <typename State, typename Input, typename EstimateUncertainty,
            typename ProcessUncertainty, typename OutputUncertainty>
  inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] input_t<Input> u,
             estimate_uncertainty<EstimateUncertainty> p,
             process_uncertainty<ProcessUncertainty> q,
             output_uncertainty<OutputUncertainty> r) const {
    using kt = kalman_s_o_i_us_ps<State, State, Input, empty_pack, empty_pack>;
    return kt{typename kt::state{x.value},
              typename kt::estimate_uncertainty{p.value},
              typename kt::process_uncertainty{q.value},
              typename kt::output_uncertainty{r.value}};
  }

  template <typename State, typename Input, typename EstimateUncertainty,
            typename OutputUncertainty, typename ProcessUncertainty,
            typename... Us, typename... Ps>
  inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] input_t<Input> u,
             estimate_uncertainty<EstimateUncertainty> p,
             process_uncertainty<ProcessUncertainty> q,
             output_uncertainty<OutputUncertainty> r,
             [[maybe_unused]] update_types_t<Us...> uts,
             [[maybe_unused]] prediction_types_t<Ps...> pts) const {

    using kt =
        kalman_s_o_i_us_ps<State, State, Input, repack_t<update_types_t<Us...>>,
                           repack_t<prediction_types_t<Ps...>>>;
    return kt{typename kt::state{x.value},
              typename kt::estimate_uncertainty{p.value},
              typename kt::process_uncertainty{q.value},
              typename kt::output_uncertainty{r.value}};
  }
};

template <typename Filter> inline constexpr make_filter<Filter> filter{};

template <typename... Arguments>
using filter_t = std::invoke_result_t<make_filter<>, Arguments...>;

} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_KALMAN_HPP
