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

#ifndef FCAROUGE_KALMAN_FILTER_INTERNAL_X_Z_P_QQ_R_HH_FF_TT_OO_US_HPP
#define FCAROUGE_KALMAN_FILTER_INTERNAL_X_Z_P_QQ_R_HH_FF_TT_OO_US_HPP

#include "policy.hpp"
#include "utility.hpp"

#include <tuple>

namespace fcarouge::kalman_filter::internal {
// Helper template to support multiple pack deduction.
template <typename, typename, typename, typename, typename, typename, typename,
          typename, typename = policies<>>
struct x_z_p_qq_r_hh_ff_tt_oo_us final {};

// The extended filter linearized about the state estimate entering each stage:
// the output model H is computed from the state and the update arguments, the
// state transition F and the process uncertainty Q from the state.
template <typename State, typename Output, typename... UpdateTypes,
          typename NoiseProcess, typename ObservationState,
          typename TransitionState, typename Transition, typename Observation,
          typename Policies>
struct x_z_p_qq_r_hh_ff_tt_oo_us<
    State, Output, std::tuple<UpdateTypes...>, NoiseProcess, ObservationState,
    TransitionState, Transition, Observation, Policies> {
  using state = State;
  using output = Output;
  using estimate_uncertainty = ᴀʙᵀ<state, state>;
  using process_uncertainty = ᴀʙᵀ<state, state>;
  using output_uncertainty = ᴀʙᵀ<output, output>;
  using state_transition = evaluate<quotient<state, state>>;
  using output_model = evaluate<quotient<output, state>>;
  using innovation = output;
  using innovation_uncertainty = output_uncertainty;
  using noise_process_function = NoiseProcess;
  using observation_state_function = ObservationState;
  using transition_state_function = TransitionState;
  using transition_function = Transition;
  using observation_function = Observation;
  using update_types = std::tuple<UpdateTypes...>;
  using prediction_types = std::tuple<>;
  using gain = evaluate<quotient<state, innovation>>;

  [[no_unique_address]] Policies policies{};
  state x{make_zero<state>()};
  estimate_uncertainty p{make_one<estimate_uncertainty>()};
  noise_process_function noise_process_q;
  output_uncertainty r{make_zero<output_uncertainty>()};
  observation_state_function observation_state_h;
  transition_state_function transition_state_f;
  transition_function transition;
  observation_function observation;

  process_uncertainty q{make_zero<process_uncertainty>()};
  output_model h{make_one<output_model>()};
  state_transition f{make_one<state_transition>()};
  gain k{make_one<gain>()};
  innovation y{make_zero<innovation>()};
  innovation_uncertainty s{make_one<innovation_uncertainty>()};
  output z{make_zero<output>()};
  update_types update_arguments{};

  constexpr void update(const UpdateTypes &...update_pack, const auto &output_z,
                        const auto &...outputs_z)
    requires(
        list_initializes<output, decltype(output_z), decltype(outputs_z)...>)
  {
    const auto &i{make_one<evaluate<product<gain, output_model>>>()};

    update_arguments = {update_pack...};
    z = output{output_z, outputs_z...};
    h = observation_state_h(x, update_pack...);
    s = innovation_uncertainty{h * p * t(h) + r};
    k = p * t(h) / s;
    y = z - observation(x, update_pack...);
    x = state{x + k * y};
    update_estimate_uncertainty(policies, p, i, k, h, r, s);
  }

  constexpr void predict() {
    // The state transition F is the Jacobian and Q the process uncertainty at
    // the state estimate before the prediction, computed ahead of the state
    // propagation.
    f = transition_state_f(x);
    q = noise_process_q(x);
    x = transition(x);
    p = estimate_uncertainty{f * p * t(f) + q};
    condition(policies, p);
  }
};
} // namespace fcarouge::kalman_filter::internal

#endif // FCAROUGE_KALMAN_FILTER_INTERNAL_X_Z_P_QQ_R_HH_FF_TT_OO_US_HPP
