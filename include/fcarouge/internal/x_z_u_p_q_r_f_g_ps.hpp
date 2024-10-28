/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.4.0
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

#ifndef FCAROUGE_INTERNAL_X_Z_U_P_Q_R_F_G_PS_HPP
#define FCAROUGE_INTERNAL_X_Z_U_P_Q_R_F_G_PS_HPP

#include "fcarouge/utility.hpp"
#include "function.hpp"

#include <tuple>

namespace fcarouge::internal {
// Helper template to support multiple pack deduction.
template <typename, typename, typename, typename, typename>
struct x_z_u_p_q_r_f_g_ps final {};

template <typename State, typename Output, typename Input,
          typename... UpdateTypes, typename... PredictionTypes>
struct x_z_u_p_q_r_f_g_ps<State, Output, Input, pack<UpdateTypes...>,
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
  using transition_state_function = function<state_transition(
      const state &, const input &, const PredictionTypes &...)>;
  using noise_process_function =
      function<process_uncertainty(const state &, const PredictionTypes &...)>;
  using transition_control_function =
      function<input_control(const PredictionTypes &...)>;
  using update_types = std::tuple<UpdateTypes...>;
  using prediction_types = std::tuple<PredictionTypes...>;

  static inline const auto i{identity_v<quotient<state, state>>};

  state x{zero_v<state>};
  estimate_uncertainty p{identity_v<estimate_uncertainty>};
  noise_process_function noise_process_q;
  output_uncertainty r{zero_v<output_uncertainty>};
  transition_state_function transition_state_f;
  transition_control_function transition_control_g;

  process_uncertainty q{zero_v<process_uncertainty>};
  input u{zero_v<input>};
  output_model h{identity_v<output_model>};
  state_transition f{identity_v<state_transition>};
  input_control g{identity_v<input_control>};
  gain k{identity_v<gain>};
  innovation y{zero_v<innovation>};
  innovation_uncertainty s{identity_v<innovation_uncertainty>};
  output z{zero_v<output>};
  prediction_types prediction_arguments{};
  transpose t{};

  inline constexpr void update(const auto &output_z, const auto &...outputs_z) {
    z = output{output_z, outputs_z...};
    s = h * p * t(h) + r;
    k = p * t(h) / s;
    y = z - h * x;
    x = state{x + k * y};
    p = estimate_uncertainty{(i - k * h) * p * t(i - k * h) + k * r * t(k)};
  }

  inline constexpr void predict(const PredictionTypes &...prediction_pack,
                                const auto &input_u, const auto &...inputs_u) {
    prediction_arguments = {prediction_pack...};
    u = input{input_u, inputs_u...};
    f = transition_state_f(x, u, prediction_pack...);
    q = noise_process_q(x, prediction_pack...);
    g = transition_control_g(prediction_pack...);
    x = f * x + g * u;
    p = estimate_uncertainty{f * p * t(f) + q};
  }
};
} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_X_Z_U_P_Q_R_F_G_PS_HPP