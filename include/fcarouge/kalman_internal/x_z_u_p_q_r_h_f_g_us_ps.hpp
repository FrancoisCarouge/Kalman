/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.3
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

#ifndef FCAROUGE_KALMAN_INTERNAL_X_Z_U_P_Q_R_H_F_G_US_PS_HPP
#define FCAROUGE_KALMAN_INTERNAL_X_Z_U_P_Q_R_H_F_G_US_PS_HPP

#include "function.hpp"
#include "utility.hpp"

#include <tuple>

namespace fcarouge::kalman_internal {
// Helper template to support multiple pack deduction.
template <typename, typename, typename, typename, typename>
struct x_z_u_p_q_r_h_f_g_us_ps final {};

template <typename State, typename Output, typename Input,
          typename... UpdateTypes, typename... PredictionTypes>
struct x_z_u_p_q_r_h_f_g_us_ps<State, Output, Input, std::tuple<UpdateTypes...>,
                               std::tuple<PredictionTypes...>> {
  using state = State;
  using output = Output;
  using input = Input;
  using estimate_uncertainty = ᴀʙᵀ<state, state>;
  using process_uncertainty = ᴀʙᵀ<state, state>;
  using output_uncertainty = ᴀʙᵀ<output, output>;
  using state_transition = ᴀʙᵀ<state, state>;
  using output_model = ᴀʙᵀ<output, state>;
  using input_control = ᴀʙᵀ<state, input>;
  using innovation = evaluate<difference<output, output>>;
  using innovation_uncertainty = output_uncertainty;
  using update_types = std::tuple<UpdateTypes...>;
  using prediction_types = std::tuple<PredictionTypes...>;
  using gain =
      evaluate<quotient<product<estimate_uncertainty, transpose<output_model>>,
                        innovation_uncertainty>>;

  static inline const auto i{one<ᴀʙᵀ<state, state>>};

  state x{zero<state>};
  estimate_uncertainty p{one<estimate_uncertainty>};
  process_uncertainty q{zero<process_uncertainty>};
  output_uncertainty r{zero<output_uncertainty>};
  output_model h{one<output_model>};
  state_transition f{one<state_transition>};
  input_control g{one<input_control>};
  input u{zero<input>};
  gain k{one<gain>};
  innovation y{zero<innovation>};
  innovation_uncertainty s{one<innovation_uncertainty>};
  output z{zero<output>};
  update_types update_arguments{};
  prediction_types prediction_arguments{};

  inline constexpr void update(const UpdateTypes &...update_pack,
                               const auto &output_z, const auto &...outputs_z) {
    update_arguments = {update_pack...};
    z = output{output_z, outputs_z...};
    s = h * p * t(h) + r;
    k = p * t(h) / s;
    y = z - h * x;
    x = x + k * y;
    p = (i - k * h) * p * t(i - k * h) + k * r * t(k);
  }

  //! @todo Add convertible requirements on input and output packs?
  inline constexpr void predict(const PredictionTypes &...prediction_pack,
                                const auto &input_u, const auto &...inputs_u) {
    prediction_arguments = {prediction_pack...};
    u = input{input_u, inputs_u...};
    x = f * x + g * u;
    p = f * p * t(f) + q;
  }
};
} // namespace fcarouge::kalman_internal

#endif // FCAROUGE_KALMAN_INTERNAL_X_Z_U_P_Q_R_H_F_G_US_PS_HPP
