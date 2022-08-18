/*_  __          _      __  __          _   _
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

#include <array>
#include <cstddef>
#include <functional>
#include <type_traits>

namespace fcarouge::internal
{

template <typename, std::size_t, std::size_t, std::size_t, typename, typename,
          typename, typename, typename, typename>
struct kalman {
  //! @todo Support some more specializations, all, or disable others?
};

template <typename Type, std::size_t State, std::size_t Output,
          typename Transpose, typename Symmetrize, typename Divide,
          typename Identity, typename... UpdateTypes,
          typename... PredictionTypes>
struct kalman<Type, State, Output, 0, Transpose, Symmetrize, Divide, Identity,
              pack<UpdateTypes...>, pack<PredictionTypes...>> {
  struct empty {
  };
  template <typename Row, typename Column>
  using matrix = std::decay_t<std::invoke_result_t<Divide, Row, Column>>;
  template <std::size_t Size>
  //! @todo Should we remove the dependency on `std::array`? It can be done with
  //! C-style `Type[Size]` array but may be not recommended.
  using array = std::conditional_t<
      Size == 1, Type,
      std::decay_t<std::invoke_result_t<
          Divide, std::conditional_t<Size == 1, Type, std::array<Type, Size>>,
          Type>>>;
  using value_type = Type;
  using state = array<State>;
  using output = array<Output>;
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
      std::function<output(const state &, const UpdateTypes &...arguments)>;

  //! @todo Is there a simpler way to initialize to the zero matrix?
  state x{ value_type{ 0 } * Identity().template operator()<state>() };
  estimate_uncertainty p{
    Identity().template operator()<estimate_uncertainty>()
  };
  process_uncertainty q{
    value_type{ 0 } * Identity().template operator()<process_uncertainty>()
  };
  output_uncertainty r{ value_type{ 0 } *
                        Identity().template operator()<output_uncertainty>() };
  output_model h{ Identity().template operator()<output_model>() };
  state_transition f{ Identity().template operator()<state_transition>() };
  gain k{ Identity().template operator()<gain>() };
  innovation y{ value_type{ 0 } *
                Identity().template operator()<innovation>() };
  innovation_uncertainty s{
    Identity().template operator()<innovation_uncertainty>()
  };
  output z{ value_type{ 0 } * Identity().template operator()<output>() };

  //! @todo Should we pass through the reference to the state x or have the user
  //! access it through k.x() when needed? Where does the practical/performance
  //! tradeoff leans toward? For the general case? For the specialized cases?
  //! Same question applies to other parameters.
  //! @todo Pass the arguments by universal reference?
  observation_state_function observation_state_h{
    [this](const state &x, const UpdateTypes &...arguments) -> output_model {
      static_cast<void>(x);
      (static_cast<void>(arguments), ...);
      return h;
    }
  };
  noise_observation_function noise_observation_r{
    [this](const state &x, const output &z,
           const UpdateTypes &...arguments) -> output_uncertainty {
      static_cast<void>(x);
      static_cast<void>(z);
      (static_cast<void>(arguments), ...);
      return r;
    }
  };
  transition_state_function transition_state_f{
    [this](const state &x,
           const PredictionTypes &...arguments) -> state_transition {
      static_cast<void>(x);
      (static_cast<void>(arguments), ...);
      return f;
    }
  };
  noise_process_function noise_process_q{
    [this](const state &x,
           const PredictionTypes &...arguments) -> process_uncertainty {
      static_cast<void>(x);
      (static_cast<void>(arguments), ...);
      return q;
    }
  };
  transition_function transition{
    [this](const state &x, const PredictionTypes &...arguments) -> state {
      (static_cast<void>(arguments), ...);
      return f * x;
    }
  };
  observation_function observation{
    [this](const state &x, const UpdateTypes &...arguments) -> output {
      (static_cast<void>(arguments), ...);
      return h * x;
    }
  };

  Transpose transpose;
  Divide divide;
  Symmetrize symmetrize;
  Identity identity;

  //! @todo Do we want to store i - k * h in a temporary result for reuse? Or
  //! does the compiler/linker do it for us?
  //! @todo Do we want to support extended custom y = output_difference(z,
  //! observation(x))?
  template <typename... Outputs>
  inline constexpr void update(const UpdateTypes &...arguments,
                               const Outputs &...output_z)
  {
    const auto i{ identity.template operator()<estimate_uncertainty>() };

    z = output{ output_z... };
    h = observation_state_h(x, arguments...); // x, z, args?
    r = noise_observation_r(x, z, arguments...);
    s = h * p * transpose(h) + r;
    k = divide(p * transpose(h), s);
    y = z - observation(x, arguments...);
    x = x + k * y;
    p = symmetrize(estimate_uncertainty{
        (i - k * h) * p * transpose(i - k * h) + k * r * transpose(k) });
  }

  inline constexpr void predict(const PredictionTypes &...arguments)
  {
    f = transition_state_f(x, arguments...);
    q = noise_process_q(x, arguments...);
    x = transition(x, arguments...);
    p = symmetrize(estimate_uncertainty{ f * p * transpose(f) + q });
  }

  template <typename... Outputs>
  inline constexpr void
  operator()(const PredictionTypes &...prediction_arguments,
             const UpdateTypes &...update_arguments, const Outputs &...output_z)
  {
    update(update_arguments..., output_z...);
    predict(prediction_arguments...);
  }
};

template <typename Type, std::size_t State, std::size_t Output,
          std::size_t Input, typename Transpose, typename Symmetrize,
          typename Divide, typename Identity, typename... UpdateTypes,
          typename... PredictionTypes>
struct kalman<Type, State, Output, Input, Transpose, Symmetrize, Divide,
              Identity, pack<UpdateTypes...>, pack<PredictionTypes...>> {
  template <typename Row, typename Column>
  using matrix = std::decay_t<std::invoke_result_t<Divide, Row, Column>>;
  template <std::size_t Size>
  using array = std::conditional_t<
      Size == 1, Type,
      std::decay_t<std::invoke_result_t<
          Divide, std::conditional_t<Size == 1, Type, std::array<Type, Size>>,
          Type>>>;
  using value_type = Type;
  using state = array<State>;
  using output = array<Output>;
  using input = array<Input>;
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
      const state &, const PredictionTypes &..., const input &)>;
  using noise_process_function = std::function<process_uncertainty(
      const state &, const PredictionTypes &...)>;
  using transition_control_function =
      std::function<input_control(const PredictionTypes &...)>;
  using transition_function =
      std::function<state(const state &, const PredictionTypes &...)>;
  using observation_function =
      std::function<output(const state &, const UpdateTypes &...arguments)>;

  //! @todo Is there a simpler way to initialize to the zero matrix?
  state x{ 0 * Identity().template operator()<state>() };
  estimate_uncertainty p{
    Identity().template operator()<estimate_uncertainty>()
  };
  process_uncertainty q{
    0 * Identity().template operator()<process_uncertainty>()
  };
  output_uncertainty r{ value_type{ 0 } *
                        Identity().template operator()<output_uncertainty>() };
  output_model h{ Identity().template operator()<output_model>() };
  state_transition f{ Identity().template operator()<state_transition>() };
  input_control g{ Identity().template operator()<input_control>() };
  gain k{ Identity().template operator()<gain>() };
  innovation y{ value_type{ 0 } *
                Identity().template operator()<innovation>() };
  innovation_uncertainty s{
    Identity().template operator()<innovation_uncertainty>()
  };
  output z{ value_type{ 0 } * Identity().template operator()<output>() };
  input u{ value_type{ 0 } * Identity().template operator()<input>() };

  //! @todo Should we pass through the reference to the state x or have the user
  //! access it through k.x() when needed? Where does the practical/performance
  //! tradeoff leans toward? For the general case? For the specialized cases?
  //! Same question applies to other parameters.
  //! @todo Pass the arguments by universal reference?
  observation_state_function observation_state_h{
    [this](const state &x, const UpdateTypes &...arguments) -> output_model {
      static_cast<void>(x);
      (static_cast<void>(arguments), ...);
      return h;
    }
  };
  noise_observation_function noise_observation_r{
    [this](const state &x, const output &z,
           const UpdateTypes &...arguments) -> output_uncertainty {
      static_cast<void>(x);
      static_cast<void>(z);
      (static_cast<void>(arguments), ...);
      return r;
    }
  };
  transition_state_function transition_state_f{
    [this](const state &x, const PredictionTypes &...arguments,
           const input &u) -> state_transition {
      static_cast<void>(x);
      (static_cast<void>(arguments), ...);
      static_cast<void>(u);
      return f;
    }
  };
  noise_process_function noise_process_q{
    [this](const state &x,
           const PredictionTypes &...arguments) -> process_uncertainty {
      static_cast<void>(x);
      (static_cast<void>(arguments), ...);
      return q;
    }
  };
  transition_control_function transition_control_g{
    [this](const PredictionTypes &...arguments) -> input_control {
      (static_cast<void>(arguments), ...);
      return g;
    }
  };
  transition_function transition{
    [this](const state &x, const PredictionTypes &...arguments) -> state {
      (static_cast<void>(arguments), ...);
      return f * x;
    }
  };
  observation_function observation{
    [this](const state &x, const UpdateTypes &...arguments) -> output {
      (static_cast<void>(arguments), ...);
      return h * x;
    }
  };

  Transpose transpose;
  Divide divide;
  Symmetrize symmetrize;
  Identity identity;

  //! @todo Do we want to store i - k * h in a temporary result for reuse? Or
  //! does the compiler/linker do it for us?
  //! @todo Do we want to support extended custom y = output_difference(z,
  //! observation(x))?
  template <typename... Outputs>
  inline constexpr void update(const UpdateTypes &...arguments,
                               const Outputs &...output_z)
  {
    const auto i{ identity.template operator()<estimate_uncertainty>() };

    z = output{ output_z... };
    h = observation_state_h(x, arguments...); // x, z, args?
    r = noise_observation_r(x, z, arguments...);
    s = h * p * transpose(h) + r;
    k = divide(p * transpose(h), s);
    y = z - observation(x, arguments...);
    x = x + k * y;
    p = symmetrize(estimate_uncertainty{
        (i - k * h) * p * transpose(i - k * h) + k * r * transpose(k) });
  }

  //! @todo Extended support?
  //! @todo Should the transition state F computation arguments be  {x, u, args}
  //! instead of {x, args, u} or can we benefit for allowing passing through an
  //! input pack to the function?
  //! @todo Should input U be passed to noise process Q compute? Probably?
  //! @todo How to extended next state x = f * x + g * u?
  template <typename... Inputs>
  inline constexpr void predict(const PredictionTypes &...arguments,
                                const Inputs &...input_u)
  {
    u = input{ input_u... };
    f = transition_state_f(x, arguments..., u);
    q = noise_process_q(x, arguments...);
    g = transition_control_g(arguments...);
    x = f * x + g * u;
    p = symmetrize(estimate_uncertainty{ f * p * transpose(f) + q });
  }

  template <typename... Inputs>
  inline constexpr void
  operator()(const PredictionTypes &...prediction_arguments,
             const UpdateTypes &...update_arguments, const Inputs &...input_u,
             const auto &...output_z)
  {
    update(update_arguments..., output_z...);
    predict(prediction_arguments..., input_u...);
  }
};

} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_KALMAN_HPP
