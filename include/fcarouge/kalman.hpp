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

#ifndef FCAROUGE_KALMAN_HPP
#define FCAROUGE_KALMAN_HPP

//! @file
//! @brief The main Kalman filter class.

#include "kalman_equation.hpp"
#include "kalman_operator.hpp"

#include <type_traits>

namespace fcarouge
{
//! @brief Kalman filter.
//!
//! @tparam State The type template parameter of the state vector x.
//! @tparam Output The type template parameter of the measurement vector z.
//! @tparam Input The type template parameter of the control u.
//! @tparam Transpose The template template parameter of the transpose functor.
//! @tparam Divide The template template parameter of the division functor.
//! @tparam Identity The template template parameter of the identity functor.
//! @tparam PredictionArguments The variadic type template parameter for
//! additional prediction function parameters. Time, or a delta thereof, is
//! often a prediction parameter.
template <typename State, typename Output = State, typename Input = State,
          template <typename> typename Transpose = transpose,
          template <typename> typename Symmetrize = symmetrize,
          template <typename, typename> typename Divide = divide,
          template <typename> typename Identity = identity,
          typename... PredictionArguments>
class kalman
{
  public:
  using state = State;
  using output = Output;
  using input = Input;
  using estimate_uncertainty =
      std::invoke_result_t<Divide<State, State>, State, State>;
  using process_noise_uncertainty =
      std::invoke_result_t<Divide<State, State>, State, State>;
  using state_transition =
      std::invoke_result_t<Divide<State, State>, State, State>;
  using observation =
      std::invoke_result_t<Divide<Output, State>, Output, State>;
  using observation_noise_uncertainty =
      std::invoke_result_t<Divide<Output, Output>, Output, Output>;
  using control = std::invoke_result_t<Divide<State, Input>, State, Input>;

  state state_x;
  estimate_uncertainty estimate_uncertainty_p;

  // Functors could be replaced by the standard general-purpose polymorphic
  // function wrapper `std::function` if lambda captures are needed.
  observation (*transition_observation_h)();
  observation_noise_uncertainty (*noise_observation_r)();

  state_transition (*transition_state_f)(const PredictionArguments &...);
  process_noise_uncertainty (*noise_process_q)(const PredictionArguments &...);
  control (*transition_control_g)(const PredictionArguments &...);

  inline constexpr void update(const output &output_z)
  {
    const auto h{ transition_observation_h() };
    const auto r{ noise_observation_r() };
    fcarouge::update<Transpose, Symmetrize, Divide, Identity>(
        state_x, estimate_uncertainty_p, h, r, output_z);
  }

  inline constexpr void predict(const PredictionArguments &...arguments)
  {
    const auto f{ transition_state_f(arguments...) };
    const auto q{ noise_process_q(arguments...) };
    fcarouge::predict<Transpose, Symmetrize>(state_x, estimate_uncertainty_p, f,
                                             q);
  }

  inline constexpr void predict(const input &input_u,
                                const PredictionArguments &...arguments)
  {
    const auto f{ transition_state_f(arguments...) };
    const auto q{ noise_process_q(arguments...) };
    const auto g{ transition_control_g(arguments...) };
    fcarouge::predict<Transpose, Symmetrize>(state_x, estimate_uncertainty_p, f,
                                             q, g, input_u);
  }
};

} // namespace fcarouge

#endif // FCAROUGE_KALMAN_HPP
