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

#ifndef FCAROUGE_KALMAN_INTERNAL_X_Z_P_Q_R_HH_US_PS_HPP
#define FCAROUGE_KALMAN_INTERNAL_X_Z_P_Q_R_HH_US_PS_HPP

#include "function.hpp"
#include "utility.hpp"

#include <tuple>

namespace fcarouge::kalman_internal {
// Helper template to support multiple pack deduction.
template <typename, typename, typename, typename>
struct x_z_p_q_r_hh_us_ps final {};

template <typename State, typename Output, typename... UpdateTypes,
          typename... PredictionTypes>
struct x_z_p_q_r_hh_us_ps<State, Output, std::tuple<UpdateTypes...>,
                          std::tuple<PredictionTypes...>> {
  using state = State;
  using output = Output;
  using estimate_uncertainty = ᴀʙᵀ<state, state>;
  using process_uncertainty = ᴀʙᵀ<state, state>;
  using output_uncertainty = ᴀʙᵀ<output, output>;
  using state_transition = ᴀʙᵀ<state, state>;
  using output_model = ᴀʙᵀ<output, state>;
  using innovation = evaluate<difference<output, output>>;
  using innovation_uncertainty = output_uncertainty;
  using observation_state_function =
      function<output_model(const state &, const UpdateTypes &...)>;
  using transition_function =
      function<state(const state &, const PredictionTypes &...)>;
  using observation_function =
      function<output(const state &, const UpdateTypes &...)>;
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
  observation_state_function observation_state_h{
      [&hh = h]([[maybe_unused]] const auto &...arguments) -> output_model {
        return hh;
      }};
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

  output_model h{one<output_model>};
  state_transition f{one<state_transition>};
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
    h = observation_state_h(x, update_pack...);
    s = innovation_uncertainty{h * p * t(h) + r};
    k = p * t(h) / s;
    y = z - observation(x, update_pack...);
    x = state{x + k * y};
    p = estimate_uncertainty{(i - k * h) * p * t(i - k * h) + k * r * t(k)};
  }

  inline constexpr void predict(const PredictionTypes &...prediction_pack) {
    prediction_arguments = {prediction_pack...};
    x = transition(x, prediction_pack...);
    p = estimate_uncertainty{f * p * t(f) + q};
  }
};
} // namespace fcarouge::kalman_internal

#endif // FCAROUGE_KALMAN_INTERNAL_X_Z_P_Q_R_HH_US_PS_HPP
