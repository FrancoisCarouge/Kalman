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

#ifndef FCAROUGE_INTERNAL_FACTORY_HPP
#define FCAROUGE_INTERNAL_FACTORY_HPP

#include "fcarouge/internal/type.hpp"
#include "fcarouge/internal/x_z_p_q_r_h_f.hpp"
#include "fcarouge/internal/x_z_p_q_r_hh_us_ps.hpp"
#include "fcarouge/internal/x_z_p_qq_rr_f.hpp"
#include "fcarouge/internal/x_z_p_r_f.hpp"
#include "fcarouge/internal/x_z_u_p_q_r_f_g_ps.hpp"
#include "fcarouge/internal/x_z_u_p_q_r_us_ps.hpp"

#include <concepts>
#include <type_traits>

namespace fcarouge::internal {
//! @todo Support arbritary order of configuration parameters?
//! @todo Support user defined types by type name reflection? Case, naming
//! convention insensitive?
//! @todo Some of these overload should probably be removed to guide the user
//! towards better initialization practices?
template <typename Filter = void> struct filter_deducer {
  template <typename... Arguments>
  static inline constexpr auto
  operator()([[maybe_unused]] Arguments... arguments)
    requires(std::same_as<Filter, void>)
  {
    static_assert(false,
                  "This requested filter configuration is not yet supported. "
                  "Please, submit a pull request or feature request.");

    return x_z_p_r_f<double, double>{};
  }

  //! @todo Rename, clean the concet: undeduced?
  static inline constexpr auto operator()() -> x_z_p_r_f<double, double>
    requires(std::same_as<Filter, void>);

  static inline constexpr auto operator()() { return Filter{}; }

  template <typename State, typename Output>
  static inline constexpr auto operator()(state<State> x,
                                          [[maybe_unused]] output_t<Output> z) {
    return x_z_p_r_f<State, Output>{x.value};
  }

  template <typename State, typename Output, typename Input>
  static inline constexpr auto operator()(state<State> x,
                                          [[maybe_unused]] output_t<Output> z,
                                          [[maybe_unused]] input_t<Input> u) {
    using kt = x_z_u_p_q_r_us_ps<State, Output, Input, empty_pack, empty_pack>;
    return kt{typename kt::state{x.value}};
  }

  template <typename State, typename Output, typename Input,
            typename EstimateUncertainty, typename ProcessUncertainty,
            typename OutputUncertainty, typename StateTransition,
            typename InputControl, typename... Ps>
  //! @todo Simplify the require clause?
  //! @todo Add clauses for StateTransition and InputControl?
    requires requires(ProcessUncertainty q) {
      requires std::invocable<ProcessUncertainty, State, Ps...>;
    }
  static inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] output_t<Output> z,
             [[maybe_unused]] input_t<Input> u,
             estimate_uncertainty<EstimateUncertainty> p,
             process_uncertainty<ProcessUncertainty> q,
             output_uncertainty<OutputUncertainty> r,
             state_transition<StateTransition> f, input_control<InputControl> g,
             [[maybe_unused]] prediction_types_t<Ps...> pts) {
    using kt = x_z_u_p_q_r_f_g_ps<State, Output, Input, empty_pack,
                                  repack_t<prediction_types_t<Ps...>>>;
    return kt{typename kt::state{x.value},
              typename kt::estimate_uncertainty{p.value},
              typename kt::noise_process_function{q.value},
              typename kt::output_uncertainty{r.value},
              typename kt::transition_state_function{f.value},
              typename kt::transition_control_function{g.value}};
  }

  template <typename State, typename Output, typename Input, typename... Us,
            typename... Ps>
  static inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] output_t<Output> z,
             [[maybe_unused]] input_t<Input> u,
             [[maybe_unused]] update_types_t<Us...> uts,
             [[maybe_unused]] prediction_types_t<Ps...> pts) {
    using kt =
        x_z_u_p_q_r_us_ps<State, Output, Input, repack_t<update_types_t<Us...>>,
                          repack_t<prediction_types_t<Ps...>>>;
    return kt{typename kt::state{x.value}};
  }

  template <typename State, typename Output, typename EstimateUncertainty,
            typename ProcessUncertainty, typename OutputUncertainty,
            typename OutputModel, typename Transition, typename Observation,
            typename... Us, typename... Ps>
  //! @todo Simplify the require clause?
    requires requires() {
      requires std::invocable<OutputModel, State, Us...>;
      requires std::invocable<Transition, State, Ps...>;
      requires std::invocable<Observation, State, Ps...>;
    }
  static inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] output_t<Output> z,
             estimate_uncertainty<EstimateUncertainty> p,
             process_uncertainty<ProcessUncertainty> q,
             output_uncertainty<OutputUncertainty> r,
             output_model<OutputModel> h, transition<Transition> ff,
             observation<Observation> hh,
             [[maybe_unused]] update_types_t<Us...> uts,
             [[maybe_unused]] prediction_types_t<Ps...> pts) {
    using kt =
        x_z_p_q_r_hh_us_ps<State, Output, repack_t<update_types_t<Us...>>,
                           repack_t<prediction_types_t<Ps...>>>;
    return kt{typename kt::state{x.value},
              typename kt::estimate_uncertainty{p.value},
              typename kt::process_uncertainty{q.value},
              typename kt::output_uncertainty{r.value},
              typename kt::observation_state_function{h.value},
              typename kt::transition_function{ff.value},
              typename kt::observation_function{hh.value}};
  }

  template <typename State, typename Output, typename EstimateUncertainty,
            typename ProcessUncertainty, typename OutputUncertainty,
            typename OutputModel, typename StateTransition>
  static inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] output_t<Output> z,
             estimate_uncertainty<EstimateUncertainty> p,
             process_uncertainty<ProcessUncertainty> q,
             output_uncertainty<OutputUncertainty> r,
             output_model<OutputModel> h, state_transition<StateTransition> f) {
    using kt = x_z_p_q_r_h_f<State, Output>;
    return kt{typename kt::state{x.value},
              typename kt::estimate_uncertainty{p.value},
              typename kt::process_uncertainty{q.value},
              typename kt::output_uncertainty{r.value},
              typename kt::output_model{h.value},
              typename kt::state_transition{f.value}};
  }

  template <typename State, typename Output, typename EstimateUncertainty,
            typename OutputUncertainty>
  static inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] output_t<Output> z,
             estimate_uncertainty<EstimateUncertainty> p,
             output_uncertainty<OutputUncertainty> r) {
    using kt = x_z_p_r_f<State, Output>;
    return kt{typename kt::state{x.value},
              typename kt::estimate_uncertainty{p.value},
              typename kt::output_uncertainty{r.value}};
  }

  template <typename State, typename Output, typename EstimateUncertainty,
            typename OutputUncertainty, typename ProcessUncertainty>
  static inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] output_t<Output> z,
             estimate_uncertainty<EstimateUncertainty> p,
             process_uncertainty<ProcessUncertainty> q,
             output_uncertainty<OutputUncertainty> r) {
    using kt = x_z_p_q_r_h_f<State, Output>;
    return kt{typename kt::state{x.value},
              typename kt::estimate_uncertainty{p.value},
              typename kt::process_uncertainty{q.value},
              typename kt::output_uncertainty{r.value}};
  }

  template <typename State, typename Output, typename Input,
            typename EstimateUncertainty, typename ProcessUncertainty,
            typename OutputUncertainty>
  static inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] output_t<Output> z,
             [[maybe_unused]] input_t<Input> u,
             estimate_uncertainty<EstimateUncertainty> p,
             process_uncertainty<ProcessUncertainty> q,
             output_uncertainty<OutputUncertainty> r) {
    using kt = x_z_u_p_q_r_us_ps<State, Output, Input, empty_pack, empty_pack>;
    return kt{typename kt::state{x.value},
              typename kt::estimate_uncertainty{p.value},
              typename kt::process_uncertainty{q.value},
              typename kt::output_uncertainty{r.value}};
  }

  template <typename State, typename Output, typename Input,
            typename EstimateUncertainty, typename OutputUncertainty,
            typename ProcessUncertainty, typename... Us, typename... Ps>
  static inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] output_t<Output> z,
             [[maybe_unused]] input_t<Input> u,
             estimate_uncertainty<EstimateUncertainty> p,
             process_uncertainty<ProcessUncertainty> q,
             output_uncertainty<OutputUncertainty> r,
             [[maybe_unused]] update_types_t<Us...> uts,
             [[maybe_unused]] prediction_types_t<Ps...> pts) {

    using kt =
        x_z_u_p_q_r_us_ps<State, Output, Input, repack_t<update_types_t<Us...>>,
                          repack_t<prediction_types_t<Ps...>>>;
    return kt{typename kt::state{x.value},
              typename kt::estimate_uncertainty{p.value},
              typename kt::process_uncertainty{q.value},
              typename kt::output_uncertainty{r.value}};
  }

  template <typename State, typename Output, typename EstimateUncertainty,
            typename ProcessUncertainty, typename OutputUncertainty,
            typename StateTransition>
  //! @todo Simplify the require clause?
    requires requires(ProcessUncertainty q, OutputUncertainty r) {
      requires std::invocable<ProcessUncertainty, State>;
      requires std::invocable<OutputUncertainty, State, Output>;
    }
  static inline constexpr auto
  operator()(state<State> x, [[maybe_unused]] output_t<Output> z,
             estimate_uncertainty<EstimateUncertainty> p,
             process_uncertainty<ProcessUncertainty> q,
             output_uncertainty<OutputUncertainty> r,
             state_transition<StateTransition> f) {
    using kt = x_z_p_qq_rr_f<State, Output>;
    return kt{typename kt::state{x.value},
              typename kt::estimate_uncertainty{p.value},
              typename kt::noise_process_function{q.value},
              typename kt::noise_observation_function{r.value},
              typename kt::state_transition{f.value}};
  }
};

template <typename Filter>
inline constexpr filter_deducer<Filter> make_filter{};

template <typename... Arguments>
using deduce_filter = std::invoke_result_t<filter_deducer<>, Arguments...>;
} // namespace fcarouge::internal

#endif // FCAROUGE_INTERNAL_FACTORY_HPP
