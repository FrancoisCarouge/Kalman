/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.4
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

#ifndef FCAROUGE_KALMAN_FILTER_INTERNAL_X_Z_U_P_Q_R_H_F_G_HPP
#define FCAROUGE_KALMAN_FILTER_INTERNAL_X_Z_U_P_Q_R_H_F_G_HPP

#include "utility.hpp"

#include <tuple>

namespace fcarouge::kalman_filter::internal {
template <typename State, typename Output, typename Input>
struct x_z_u_p_q_r_h_f_g {
  using state = State;
  using output = Output;
  using input = Input;
  using estimate_uncertainty = ᴀʙᵀ<state, state>;
  using process_uncertainty = ᴀʙᵀ<state, state>;
  using output_uncertainty = ᴀʙᵀ<output, output>;
  using state_transition = evaluate<quotient<state, state>>;
  using output_model = evaluate<quotient<output, state>>;
  using input_control = evaluate<quotient<state, input>>;
  using innovation = evaluate<difference<output, output>>;
  using innovation_uncertainty = output_uncertainty;
  using update_types = std::tuple<>;
  using prediction_types = std::tuple<>;
  using gain = evaluate<quotient<state, innovation>>;

  state x{make_zero<state>()};
  estimate_uncertainty p{make_one<estimate_uncertainty>()};
  process_uncertainty q{make_zero<process_uncertainty>()};
  output_uncertainty r{make_zero<output_uncertainty>()};
  output_model h{make_one<output_model>()};
  state_transition f{make_one<state_transition>()};
  input_control g{make_one<input_control>()};
  input u{make_zero<input>()};
  gain k{make_one<gain>()};
  innovation y{make_zero<innovation>()};
  innovation_uncertainty s{make_one<innovation_uncertainty>()};
  output z{make_zero<output>()};

  constexpr void update(const auto &output_z, const auto &...outputs_z)
    requires(
        list_initializes<output, decltype(output_z), decltype(outputs_z)...>)
  {
    const auto i{make_one<evaluate<product<gain, output_model>>>()};
    z = output{output_z, outputs_z...};
    s = h * p * t(h) + r;
    k = p * t(h) / s;
    y = z - h * x;
    x = x + k * y;
    p = (i - k * h) * p * t(i - k * h) + k * r * t(k);
  }

  //! @todo Add convertible requirements on input and output packs?
  constexpr void predict(const auto &input_u, const auto &...inputs_u)
    requires(list_initializes<input, decltype(input_u), decltype(inputs_u)...>)
  {
    u = input{input_u, inputs_u...};
    x = f * x + g * u;
    p = f * p * t(f) + q;
  }
};
} // namespace fcarouge::kalman_filter::internal

#endif // FCAROUGE_KALMAN_FILTER_INTERNAL_X_Z_U_P_Q_R_H_F_G_HPP
