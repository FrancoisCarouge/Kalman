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

//! @file
//! @brief The main Kalman filter class.

#include <functional>
#include <type_traits>

namespace fcarouge::internal
{
template <typename State, typename Output, typename Input, typename Transpose,
          typename Symmetrize, typename Divide, typename Identity,
          typename... PredictionArguments>
struct kalman {
  //! @name Public Member Types
  //! @{

  //! @brief Type of the state estimate vector X.
  using state = State;

  //! @brief Type of the observation vector Z.
  //!
  //! @details Also known as Y or O.
  using output = Output;

  //! @brief Type of the control vector U.
  using input = Input;

  //! @brief Type of the estimated covariance matrix P.
  //!
  //! @details Also known as Σ.
  using estimate_uncertainty =
      std::decay_t<std::invoke_result_t<Divide, State, State>>;

  //! @brief Type of the process noise covariance matrix Q.
  using process_uncertainty =
      std::decay_t<std::invoke_result_t<Divide, State, State>>;

  //! @brief Type of the observation, measurement noise covariance matrix R.
  using output_uncertainty =
      std::decay_t<std::invoke_result_t<Divide, Output, Output>>;

  //! @brief Type of the state transition matrix F.
  //!
  //! @details Also known as Φ or A.
  using state_transition =
      std::decay_t<std::invoke_result_t<Divide, State, State>>;

  //! @brief Type of the observation transition matrix H.
  //!
  //! @details Also known as C.
  using output_model =
      std::decay_t<std::invoke_result_t<Divide, Output, State>>;

  //! @brief Type of the control transition matrix G.
  //!
  //! @details Also known as B.
  using input_control =
      std::decay_t<std::invoke_result_t<Divide, State, Input>>;

  //! @brief Type of the gain matrix K.
  using gain = std::decay_t<std::invoke_result_t<Transpose, output_model>>;

  //! @brief Type of the innovation vector Y.
  using innovation = output;

  //! @brief Type of the innovation uncertainty matrix S.
  using innovation_uncertainty = output_uncertainty;

  //! @}

  //! @name Public Member Variables
  //! @{

  //! @brief The state estimate vector x.
  //!
  //! @todo Is there a simpler, more portable way to get a zero initialization?
  state x{ 0 * Identity().template operator()<state>() };

  //! @brief The estimate uncertainty, covariance matrix P.
  //!
  //! @details The estimate uncertainty, covariance is also known as Σ.
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

  //! @}

  //! @name Public Member Function Objects
  //! @{

  //! @brief Compute the state observation H matrix.
  //!
  //! @details The state observation H is also known as C.
  //! For non-linear system, or extended filter, H is the Jacobian of the state
  //! observation function. H = ∂h/∂X = ∂hj/∂xi that is each row i
  //! contains the derivatives of the state observation function for every
  //! element j in the state vector X.
  std::function<output_model(const state &)> observation_state_h{
    [this](const state &x) -> output_model {
      static_cast<void>(x);
      return h;
    }
  };

  //! @brief Compute observation noise R matrix.
  std::function<output_uncertainty()> noise_observation_r{
    [this] -> output_uncertainty { return r; }
  };

  //! @brief Compute the state transition F matrix.
  //!
  //! @details The state transition F matrix is also known as Φ or A.
  //! For non-linear system, or extended filter, F is the Jacobian of the state
  //! transition function. F = ∂f/∂X = ∂fj/∂xi that is each row i contains the
  //! derivatives of the state transition function for every element j in the
  //! state vector X.
  //!
  //! @todo Pass the arguments by universal reference?
  std::function<state_transition(const PredictionArguments &...)>
      transition_state_f{
        [this](const PredictionArguments &...arguments) -> state_transition {
          (static_cast<void>(arguments), ...);
          return f;
        }
      };

  //! @brief Compute process noise Q matrix.
  std::function<process_uncertainty(const PredictionArguments &...)>
      noise_process_q{
        [this](const PredictionArguments &...arguments) -> process_uncertainty {
          (static_cast<void>(arguments), ...);
          return q;
        }
      };

  //! @brief Compute control transition G matrix.
  std::function<input_control(const PredictionArguments &...)>
      transition_control_g{
        [this](const PredictionArguments &...arguments) -> input_control {
          (static_cast<void>(arguments), ...);
          return g;
        }
      };

  //! @brief State transition function f.
  //!
  //! @details For linear system f(x) = F * X. For non-linear system, or
  //! extended filter, the client implements a linearization of the transition
  //! function f and the state transition F matrix is the Jacobian of the state
  //! transition function.
  std::function<state(const state &, const PredictionArguments &...)>
      transition{ [this](const state &x,
                         const PredictionArguments &...arguments) -> state {
        (static_cast<void>(arguments), ...);
        return f * x;
      } };

  //! @brief State observation function h.
  //!
  //! @details For linear system h(x) = H * X. For non-linear system, or
  //! extended filter, the client implements a linearization of the observation
  //! function hand the state observation H matrix is the Jacobian of the state
  //! observation function.
  std::function<output(const state &)> observation{
    [this](const state &x) -> output { return h * x; }
  };

  Transpose transpose;
  Divide divide;
  Symmetrize symmetrize;
  Identity identity;

  //! @}

  //! @name Public Member Functions
  //! @{

  //! @todo Do we want to store i - k * h in a temporary result for reuse? Or
  //! does the compiler/linker do it for us?
  //! @todo H would be the observe Jacobian(x) extended?
  //! @todo Would innovation y = z - extended_hh(x) be extended?
  inline constexpr void update(const auto &...output_z)
  {
    const auto i{ identity.template operator()<estimate_uncertainty>() };

    z = output{ output_z... };
    h = observation_state_h(x);
    r = noise_observation_r();
    s = h * p * transpose(h) + r;
    k = divide(p * transpose(h), s);
    y = z - observation(x);
    x = x + k * y;
    p = symmetrize(estimate_uncertainty{
        (i - k * h) * p * transpose(i - k * h) + k * r * transpose(k) });
  }

  //! @todo F would be the predict Jacobian(x) extended?
  //! @todo Would x = extended_ff(x, u) be extended?
  inline constexpr void predict(const PredictionArguments &...arguments,
                                const auto &...input_u)
  {
    const auto u{ input{ input_u... } };

    f = transition_state_f(arguments...);
    q = noise_process_q(arguments...);
    g = transition_control_g(arguments...);

    x = f * x + g * u;
    p = symmetrize(estimate_uncertainty{ f * p * transpose(f) + q });
  }

  inline constexpr void predict(const PredictionArguments &...arguments)
  {
    f = transition_state_f(arguments...);
    q = noise_process_q(arguments...);

    x = transition(x, arguments...);
    p = symmetrize(estimate_uncertainty{ f * p * transpose(f) + q });
  }

  //! @}
};

} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_KALMAN_HPP
