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

#include "kalman_equation.hpp"
#include "kalman_operator.hpp"

#include <functional>
#include <type_traits>

namespace fcarouge::internal
{
template <typename State, typename Output = State, typename Input = State,
          template <typename> typename Transpose = transpose,
          template <typename> typename Symmetrize = symmetrize,
          template <typename, typename> typename Divide = divide,
          template <typename> typename Identity = identity,
          typename... PredictionArguments>
struct kalman {
  //! @name Public Member Types
  //! @{

  //! @brief Type of the state estimate vector X.
  using state = State;

  //! @brief Type of the observation vector Z.
  //!
  //! @details Also known as Y.
  using output = Output;

  //! @brief Type of the control vector U.
  using input = Input;

  //! @brief Type of the estimated covariance matrix P.
  //!
  //! @details Also known as Σ.
  using estimate_uncertainty =
      std::invoke_result_t<Divide<State, State>, State, State>;

  //! @brief Type of the process noise covariance matrix Q.
  using process_uncertainty =
      std::invoke_result_t<Divide<State, State>, State, State>;

  //! @brief Type of the observation, measurement noise covariance matrix R.
  using output_uncertainty =
      std::invoke_result_t<Divide<Output, Output>, Output, Output>;

  //! @brief Type of the state transition matrix F.
  //!
  //! @details Also known as Φ or A.
  using state_transition =
      std::invoke_result_t<Divide<State, State>, State, State>;

  //! @brief Type of the observation transition matrix H.
  //!
  //! @details Also known as C.
  using output_model =
      std::invoke_result_t<Divide<Output, State>, Output, State>;

  //! @brief Type of the control transition matrix G.
  //!
  //! @details Also known as B.
  using input_control =
      std::invoke_result_t<Divide<State, Input>, State, Input>;

  //! @}

  //! @name Public Member Variables
  //! @{

  //! @brief The state estimate vector x.
  state x{ 0 * Identity<state>()() };

  //! @brief The estimate uncertainty, covariance matrix P.
  //!
  //! @details The estimate uncertainty, covariance is also known as Σ.
  estimate_uncertainty p{ Identity<estimate_uncertainty>()() };

  process_uncertainty q{ 0 * Identity<process_uncertainty>()() };
  output_uncertainty r{ 0 * Identity<output_uncertainty>()() };
  output_model h{ Identity<output_model>()() };
  state_transition f{ Identity<state_transition>()() };
  input_control g{ Identity<input_control>()() };

  //! @}

  //! @name Public Member Function Objects
  //! @{

  //! @brief Compute observation transition H matrix.
  //!
  //! @details The observation transition H is also known as C.
  std::function<output_model()> transition_observation_h{ [this] {
    return h;
  } };

  //! @brief Compute observation noise R matrix.
  std::function<output_uncertainty()> noise_observation_r{ [this] {
    return r;
  } };

  //! @brief Compute state transition F matrix.
  //!
  //! @details The state transition F matrix is also known as Φ or A.
  //! For non-linear system, or extended filter, F is the Jacobian of the state
  //! transition function. F = ∂fj/∂xi that is each row i contains the the
  //! derivatives of the state transition function for every element j in the
  //! state vector x.
  std::function<state_transition(const PredictionArguments &...)>
      transition_state_f{ [this](const PredictionArguments &...arguments) {
        static_cast<void>((arguments, ...));
        return f;
      } };

  //! @brief Compute process noise Q matrix.
  std::function<process_uncertainty(const PredictionArguments &...)>
      noise_process_q{ [this](const PredictionArguments &...arguments) {
        static_cast<void>((arguments, ...));
        return q;
      } };

  //! @brief Compute control transition G matrix.
  std::function<input_control(const PredictionArguments &...)>
      transition_control_g{ [this](const PredictionArguments &...arguments) {
        static_cast<void>((arguments, ...));
        return g;
      } };

  //! @brief State transition function.
  //!
  //! @details
  // Add prediction arguments?
  std::function<state(const state &, const state_transition &)> predict_state =
      [](const state &x, const state_transition &f) { return state{ f * x }; };

  //! @}

  //! @name Public Member Functions
  //! @{

  inline constexpr void update(const auto &...output_z)
  {
    h = transition_observation_h();
    r = noise_observation_r();
    const auto z{ output{ output_z... } };
    internal::update<Transpose, Symmetrize, Divide, Identity>(x, p, h, r, z);
  }

  inline constexpr void predict(const PredictionArguments &...arguments,
                                const auto &...input_u)
  {
    // use member variables
    const auto ff{ predict_state };
    f = transition_state_f(arguments...);
    q = noise_process_q(arguments...);
    g = transition_control_g(arguments...);
    const auto u{ input{ input_u... } };
    internal::predict<Transpose, Symmetrize>(x, p, ff, f, q, g, u);
  }

  //! @}
};

} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_KALMAN_HPP
