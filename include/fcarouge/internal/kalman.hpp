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

#include <functional>
#include <tuple>
#include <type_traits>

namespace fcarouge::internal
{

template <typename State, typename Output, typename Input, typename Transpose,
          typename Symmetrize, typename Divide, typename Identity,
          typename UpdateArguments, typename PredictionArguments>
struct kalman {
  //! @todo Support some, all, or disable?
};

//! @todo Remove `std::tuple` dependency.
template <typename State, typename Output, typename Input, typename Transpose,
          typename Symmetrize, typename Divide, typename Identity,
          typename... UpdateArguments, typename... PredictionArguments>
struct kalman<State, Output, Input, Transpose, Symmetrize, Divide, Identity,
              std::tuple<UpdateArguments...>,
              std::tuple<PredictionArguments...>> {
  using state = State;
  using output = Output;
  using input = Input;
  using estimate_uncertainty =
      std::decay_t<std::invoke_result_t<Divide, State, State>>;
  using process_uncertainty =
      std::decay_t<std::invoke_result_t<Divide, State, State>>;
  using output_uncertainty =
      std::decay_t<std::invoke_result_t<Divide, Output, Output>>;
  using state_transition =
      std::decay_t<std::invoke_result_t<Divide, State, State>>;
  using output_model =
      std::decay_t<std::invoke_result_t<Divide, Output, State>>;
  using input_control =
      std::decay_t<std::invoke_result_t<Divide, State, Input>>;
  using gain = std::decay_t<std::invoke_result_t<Transpose, output_model>>;
  using innovation = output;
  using innovation_uncertainty = output_uncertainty;
  using observation_state_function =
      std::function<output_model(const state &, const UpdateArguments &...)>;
  using noise_observation_function = std::function<output_uncertainty(
      const state &, const output &, const UpdateArguments &...)>;
  using transition_state_function = std::function<state_transition(
      const state &, const PredictionArguments &..., const input &)>;
  using noise_process_function = std::function<process_uncertainty(
      const state &, const PredictionArguments &...)>;
  using transition_control_function =
      std::function<input_control(const PredictionArguments &...)>;
  using transition_function =
      std::function<state(const state &, const PredictionArguments &...)>;
  using observation_function =
      std::function<output(const state &, const UpdateArguments &...arguments)>;

  //! @todo Is there a simpler way to initialize to the zero matrix?
  state x{ 0 * Identity().template operator()<state>() };
  estimate_uncertainty p{
    Identity().template operator()<estimate_uncertainty>()
  };
  process_uncertainty q{
    0 * Identity().template operator()<process_uncertainty>()
  };
  output_uncertainty r{ 0 *
                        Identity().template operator()<output_uncertainty>() };
  output_model h{ Identity().template operator()<output_model>() };
  state_transition f{ Identity().template operator()<state_transition>() };
  input_control g{ Identity().template operator()<input_control>() };
  gain k{ Identity().template operator()<gain>() };
  innovation y{ 0 * Identity().template operator()<innovation>() };
  innovation_uncertainty s{
    Identity().template operator()<innovation_uncertainty>()
  };
  output z{ 0 * Identity().template operator()<output>() };
  input u{ 0 * Identity().template operator()<input>() };

  //! @todo Should we pass through the reference to the state x or have the user
  //! access it through k.x() when needed? Where does the practical/performance
  //! tradeoff leans toward? For the general case? For the specialized cases?
  //! Same question applies to other parameters.
  //! @todo Pass the arguments by universal reference?
  observation_state_function observation_state_h{
    [this](const state &x,
           const UpdateArguments &...arguments) -> output_model {
      static_cast<void>(x);
      (static_cast<void>(arguments), ...);
      return h;
    }
  };
  noise_observation_function noise_observation_r{
    [this](const state &x, const output &z,
           const UpdateArguments &...arguments) -> output_uncertainty {
      static_cast<void>(x);
      static_cast<void>(z);
      (static_cast<void>(arguments), ...);
      return r;
    }
  };
  transition_state_function transition_state_f{
    [this](const state &x, const PredictionArguments &...arguments,
           const input &u) -> state_transition {
      static_cast<void>(x);
      (static_cast<void>(arguments), ...);
      static_cast<void>(u);
      return f;
    }
  };
  noise_process_function noise_process_q{
    [this](const state &x,
           const PredictionArguments &...arguments) -> process_uncertainty {
      static_cast<void>(x);
      (static_cast<void>(arguments), ...);
      return q;
    }
  };
  transition_control_function transition_control_g{
    [this](const PredictionArguments &...arguments) -> input_control {
      (static_cast<void>(arguments), ...);
      return g;
    }
  };
  transition_function transition{
    [this](const state &x, const PredictionArguments &...arguments) -> state {
      (static_cast<void>(arguments), ...);
      return f * x;
    }
  };
  observation_function observation{
    [this](const state &x, const UpdateArguments &...arguments) -> output {
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
  inline constexpr void update(const UpdateArguments &...arguments,
                               const auto &...output_z)
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
  inline constexpr void predict(const PredictionArguments &...arguments,
                                const auto &...input_u)
  {
    u = input{ input_u... };
    f = transition_state_f(x, arguments..., u);
    q = noise_process_q(x, arguments...);
    g = transition_control_g(arguments...);
    x = f * x + g * u;
    p = symmetrize(estimate_uncertainty{ f * p * transpose(f) + q });
  }

  inline constexpr void predict(const PredictionArguments &...arguments)
  {
    f = transition_state_f(x, arguments..., input{});
    q = noise_process_q(x, arguments...);
    x = transition(x, arguments...);
    p = symmetrize(estimate_uncertainty{ f * p * transpose(f) + q });
  }

  inline constexpr void
  operator()(const PredictionArguments &...prediction_arguments,
             const UpdateArguments &...update_arguments, const auto &...input_u,
             const auto &...output_z)
  {
    update(update_arguments..., output_z...);
    predict(prediction_arguments..., input_u...);
  }
};

} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_KALMAN_HPP
