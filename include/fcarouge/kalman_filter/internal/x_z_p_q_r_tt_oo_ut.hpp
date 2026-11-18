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

#ifndef FCAROUGE_KALMAN_FILTER_INTERNAL_X_Z_P_Q_R_TT_OO_UT_HPP
#define FCAROUGE_KALMAN_FILTER_INTERNAL_X_Z_P_Q_R_TT_OO_UT_HPP

#include "policy.hpp"
#include "utility.hpp"

#include <array>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace fcarouge::kalman_filter::internal {
// The sigma points propagated by the prediction and retained for the update,
// and whether they are still to be observed.
template <typename State, std::size_t Points> struct retained_sigma_points {
  std::array<State, Points> values{};
  bool propagated{false};
};

// Nothing is retained by a filter drawing its update sigma points anew.
struct no_sigma_points {};

// The unscented filter: the state is propagated through the transition
// function, and observed through the observation function, at the 2n + 1 sigma
// points of the scaled unscented transform, without Jacobians. The noises are
// additive. The prediction draws the sigma points from the estimate and its
// uncertainty. The update draws the sigma points anew from the predicted
// estimate and its uncertainty, which account for the process uncertainty and
// the conditioning. With the reuse policy, the update observes the sigma points
// propagated by the prediction instead, retained in the filter, or draws them
// anew without a prediction since the last update.
template <typename State, typename Output, typename Transition,
          typename Observation, typename Parameter,
          typename Policies = policies<>>
struct x_z_p_q_r_tt_oo_ut {
  using state = State;
  using output = Output;
  using estimate_uncertainty = ᴀʙᵀ<state, state>;
  using process_uncertainty = ᴀʙᵀ<state, state>;
  using output_uncertainty = ᴀʙᵀ<output, output>;
  using innovation = output;
  using innovation_uncertainty = output_uncertainty;
  using transition_function = Transition;
  using observation_function = Observation;
  using update_types = std::tuple<>;
  using prediction_types = std::tuple<>;
  using gain = evaluate<quotient<state, innovation>>;
  using cross_uncertainty = ᴀʙᵀ<state, output>;
  using parameter = Parameter;

  static constexpr std::size_t dimension{dimension_v<state>};
  static constexpr std::size_t points{(2 * dimension) + 1};
  static constexpr bool reuses{contains<reuse_sigma_points_t, Policies>};

  using sigma_points = std::array<state, points>;
  using retained =
      std::conditional_t<reuses, retained_sigma_points<state, points>,
                         no_sigma_points>;

  [[no_unique_address]] Policies policies{};
  state x{make_zero<state>()};
  estimate_uncertainty p{make_one<estimate_uncertainty>()};
  process_uncertainty q{make_zero<process_uncertainty>()};
  output_uncertainty r{make_zero<output_uncertainty>()};
  transition_function transition;
  observation_function observation;
  parameter alpha{1};
  parameter beta{0};
  parameter kappa{0};

  gain k{make_one<gain>()};
  innovation y{make_zero<innovation>()};
  innovation_uncertainty s{make_one<innovation_uncertainty>()};
  output z{make_zero<output>()};
  [[no_unique_address]] retained sigma{};

  constexpr void update(const auto &output_z, const auto &...outputs_z)
    requires(
        list_initializes<output, decltype(output_z), decltype(outputs_z)...>)
  {
    z = output{output_z, outputs_z...};

    if constexpr (reuses) {
      if (not sigma.propagated) {
        draw(sigma.values);
      }

      sigma.propagated = false;
      observe(sigma.values);
    } else {
      sigma_points drawn{};

      draw(drawn);
      observe(drawn);
    }
  }

  constexpr void predict() {
    sigma_points propagated{};

    draw(propagated);

    for (auto &point : propagated) {
      point = state{transition(point)};
    }

    x = mean(propagated);
    p = estimate_uncertainty{q};

    for (std::size_t index{0}; index < points; ++index) {
      const state spread{propagated[index] - x};

      p = estimate_uncertainty{p + weight_c(index) * spread * t(spread)};
    }

    condition(policies, p);

    if constexpr (reuses) {
      sigma.values = propagated;
      sigma.propagated = true;
    }
  }

  // The update of the estimate from the observation of the sigma points.
  constexpr void observe(const sigma_points &sigma_x) {
    std::array<output, points> sigma_z{};

    for (std::size_t index{0}; index < points; ++index) {
      sigma_z[index] = output{observation(sigma_x[index])};
    }

    const output expected{mean(sigma_z)};
    s = innovation_uncertainty{r};
    cross_uncertainty pxz{make_zero<cross_uncertainty>()};

    for (std::size_t index{0}; index < points; ++index) {
      const output deviation{sigma_z[index] - expected};
      const state spread{sigma_x[index] - x};

      s = innovation_uncertainty{s +
                                 weight_c(index) * deviation * t(deviation)};
      pxz = cross_uncertainty{pxz + weight_c(index) * spread * t(deviation)};
    }

    k = pxz / s;
    y = z - expected;
    x = state{x + k * y};
    p = estimate_uncertainty{p - k * s * t(k)};
    condition(policies, p);
  }

  // The squared spread of the sigma points, α²(n + κ) = n + λ.
  [[nodiscard]] constexpr auto scale() const -> parameter {
    return alpha * alpha * (static_cast<parameter>(dimension) + kappa);
  }

  [[nodiscard]] constexpr auto weight_m(std::size_t index) const -> parameter {
    if (index == 0) {
      return 1 - (static_cast<parameter>(dimension) / scale());
    }

    return 1 / (2 * scale());
  }

  [[nodiscard]] constexpr auto weight_c(std::size_t index) const -> parameter {
    if (index == 0) {
      return weight_m(0) + 1 - (alpha * alpha) + beta;
    }

    return weight_m(index);
  }

  template <typename Type>
  [[nodiscard]] constexpr auto
  mean(const std::array<Type, points> &values) const -> Type {
    Type result{weight_m(0) * values[0]};

    for (std::size_t index{1}; index < points; ++index) {
      result = Type{result + weight_m(index) * values[index]};
    }

    return result;
  }

  // The column of the square root as a state, from its elements.
  template <std::size_t Column>
  [[nodiscard]] static constexpr auto column(const auto &root) -> state {
    return [&root]<std::size_t... Rows>(std::index_sequence<Rows...>) {
      return state{at<Rows, Column>(root)...};
    }(std::make_index_sequence<dimension>{});
  }

  // The sigma points about the estimate, x and x ± √((n + λ)P)ᵢ.
  constexpr void draw(sigma_points &sigma_x) const {
    const auto root{cholesky(estimate_uncertainty{scale() * p})};

    sigma_x[0] = x;

    [&]<std::size_t... Columns>(std::index_sequence<Columns...>) {
      ((sigma_x[1 + Columns] = state{x + column<Columns>(root)},
        sigma_x[1 + dimension + Columns] = state{x - column<Columns>(root)}),
       ...);
    }(std::make_index_sequence<dimension>{});
  }
};
} // namespace fcarouge::kalman_filter::internal

#endif // FCAROUGE_KALMAN_FILTER_INTERNAL_X_Z_P_Q_R_TT_OO_UT_HPP
