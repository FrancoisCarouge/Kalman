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

#ifndef FCAROUGE_INTERNAL_X_Z_P_Q_R_HH_US_PS_HPP
#define FCAROUGE_INTERNAL_X_Z_P_Q_R_HH_US_PS_HPP

#include "fcarouge/utility.hpp"
#include "function.hpp"

#include <tuple>

namespace fcarouge::internal {
// Helper template to support multiple pack deduction.
template <typename, typename, typename, typename>
struct x_z_p_q_r_hh_us_ps final {};

template <typename State, typename Output, typename... UpdateTypes,
          typename... PredictionTypes>
struct x_z_p_q_r_hh_us_ps<State, Output, pack<UpdateTypes...>,
                          pack<PredictionTypes...>> {
  using state = State;
  using output = Output;
  using estimate_uncertainty = deduce_matrix<state, state>;
  using process_uncertainty = deduce_matrix<state, state>;
  using output_uncertainty = deduce_matrix<output, output>;
  using state_transition = deduce_matrix<state, state>;
  using output_model = deduce_matrix<output, state>;
  using gain = deduce_matrix<state, output>;
  using innovation = output;
  using innovation_uncertainty = output_uncertainty;
  using observation_state_function =
      function<output_model(const state &, const UpdateTypes &...)>;
  using transition_function =
      function<state(const state &, const PredictionTypes &...)>;
  using observation_function =
      function<output(const state &, const UpdateTypes &...)>;
  using update_types = std::tuple<UpdateTypes...>;
  using prediction_types = std::tuple<PredictionTypes...>;

  static inline const auto i{identity_v<deduce_matrix<state, state>>};

  state x{zero_v<state>};
  estimate_uncertainty p{identity_v<estimate_uncertainty>};
  process_uncertainty q{zero_v<process_uncertainty>};
  output_uncertainty r{zero_v<output_uncertainty>};
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

  output_model h{identity_v<output_model>};
  state_transition f{identity_v<state_transition>};
  gain k{identity_v<gain>};
  innovation y{zero_v<innovation>};
  innovation_uncertainty s{identity_v<innovation_uncertainty>};
  output z{zero_v<output>};
  update_types update_arguments{};
  prediction_types prediction_arguments{};
  transposer t{};

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
} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_X_Z_P_Q_R_HH_US_PS_HPP
