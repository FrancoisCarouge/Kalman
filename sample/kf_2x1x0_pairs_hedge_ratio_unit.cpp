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

#include "fcarouge/kalman.hpp"
#include "fcarouge/linalg.hpp"

#include <cassert>

#include <mp-units/compat_macros.h>
#include <mp-units/framework/dimension.h>

namespace fcarouge::sample {
namespace {
//! @brief The currency, not a physical quantity, is a base dimension of its
//! own: a price never converts to nor compares with a physical quantity.
constexpr struct dim_currency final : mp_units::base_dimension<"$"> {
} dim_currency;
QUANTITY_SPEC(currency, dim_currency);
constexpr struct us_dollar final
    : mp_units::named_unit<"USD", mp_units::kind_of<currency>> {
} us_dollar;
constexpr auto USD{us_dollar};
constexpr auto USD2{pow<2>(USD)};

using price = mp_units::quantity<USD>;
template <typename... Types> using vector = column_vector<double, Types...>;
using ratio = mp_units::quantity<mp_units::one>;
using state_t = vector<ratio, price>;
using output_t = vector<price>;
using output_model_t = kf::evaluate<kf::quotient<output_t, state_t>>;

//! @brief Estimating the pairs trading hedge ratio.
//!
//! @copyright This example implements the dynamic linear regression of the mean
//! reversion pairs trading strategy of Ernest P. Chan, with its parameters. The
//! closing prices are simulated and the expected values computed independently.
//!
//! @see Ernest P. Chan, Algorithmic Trading: Winning Strategies and Their
//! Rationale, John Wiley & Sons, 2013, chapter 3 Implementing Mean Reversion
//! Strategies, pages 63-85, example 3.3, https://doi.org/10.1002/9781118676998
//!
//! @details The filter regresses the closing price y of a dependent stock on
//! the closing price x of an independent stock, estimating the drifting hedge
//! ratio β and intercept α. The output model H = [x, 1] changes every day:
//! - yt = βt xt + αt + εt, εt ~ N(0, Ve), Ve = 0.001 $^2.
//! - βt+1 = βt + ωβ, αt+1 = αt + ωα, ω ~ N(0, Vw), Vw = δ / (1 - δ),
//! δ = 0.0001.
//! The Vw, Ve, R, and Q of the reference are the Q, R, predicted P, and S of
//! the filter. The innovation e is the spread of the pair: the strategy enters
//! long when e < -√S and short when e > √S. Sixty daily closing prices are
//! simulated with a hedge ratio drifting from 1.25 to 1.37 and a null
//! intercept, which a small δ adapts slowly. Not trading advice.
//!
//! @image html ./sample/image/kf_2x1x0_pairs_hedge_ratio_unit.svg
//!
//! @example kf_2x1x0_pairs_hedge_ratio_unit.cpp
[[maybe_unused]] const auto sample{[] -> int {
  // A 2x1x0 filter, random walk dynamic model, no control, with the price of
  // the independent stock as an update argument.
  kalman filter{
      // The state X is the hedge ratio and intercept: [β, α]. As in the
      // reference, we start from a null hedge ratio and intercept.
      state{state_t{ratio{0. * mp_units::one}, price{0. * USD}}},
      // The filter observes the output Z dependent closing price [$].
      output<output_t>,
      // As in the reference, the initial estimate uncertainty P is null: the
      // first closing prices do not update the state, after which the process
      // uncertainty accumulates.
      estimate_uncertainty{kf::ᴀʙᵀ<state_t, state_t>{}},
      // The process uncertainty Q: Vw = δ / (1 - δ) with δ = 0.0001, for the
      // dimensionless hedge ratio and for the intercept in dollars.
      process_uncertainty{[]() -> auto {
        using process_uncertainty_t = kf::ᴀʙᵀ<state_t, state_t>;
        const double delta{0.0001};
        process_uncertainty_t value;
        value.at<0, 0>(delta / (1. - delta) * mp_units::one);
        value.at<1, 1>(delta / (1. - delta) * USD2);
        return value;
      }()},
      // The output uncertainty R: Ve = 0.001 $^2.
      output_uncertainty{0.001 * USD2},
      // The output model H = [x, 1] of the regression changes every day with
      // the independent closing price x.
      output_model{[]([[maybe_unused]] const state_t &x,
                      const price &independent) -> output_model_t {
        output_model_t value;
        value.at<0>(independent);
        value.at<1>(1.);
        return value;
      }},
      // The hedge ratio and intercept follow a random walk: their prediction
      // is their last estimate.
      transition{[](const state_t &x) -> state_t { return x; }},
      // The regression forecasts the dependent closing price: βx + α.
      observation{[](const state_t &x, const price &independent) -> output_t {
        return output_t{x.at<0>() * independent + x.at<1>()};
      }},
      // The filter update uses the independent closing price [$] parameter.
      update_types<price>,
      // The filter prediction uses no parameter.
      prediction_types<>};

  // Verifies a value at the relative accuracy of its expectation.
  const auto near{
      [](const auto &value, const auto &expected, double accuracy) -> bool {
        return abs(value - expected) < accuracy * abs(expected);
      }};

  // The first day only updates the filter, as in the reference: with a null
  // estimate uncertainty, the gain is null and the state remains unchanged.
  filter.update(20. * USD, 25. * USD);

  assert(filter.x().at<0>() == 0. * mp_units::one &&
         filter.x().at<1>() == 0. * USD && "The state remains null.");

  // Then, run a step of the filter every trading day, predicting the hedge
  // ratio and intercept of the day and updating them with the closing prices
  // of the independent and dependent stocks of the day.
  const auto step{[&filter](price independent, price dependent) -> void {
    filter.predict();
    filter.update(independent, dependent);
  }};

  step(19.94 * USD, 24.71 * USD);

  // The process uncertainty accumulated over one day lets the second closing
  // prices set the hedge ratio.
  assert(near(filter.x().at<0>(), 1.20586 * mp_units::one, 0.001) &&
         near(filter.x().at<1>(), 0.0604744 * USD, 0.001) &&
         "The state estimates expected at 0.1% accuracy.");

  step(20.19 * USD, 24.97 * USD);
  step(19.99 * USD, 24.64 * USD);
  step(19.9 * USD, 24.35 * USD);
  step(20.09 * USD, 25.03 * USD);
  step(20.27 * USD, 25.16 * USD);
  step(20.56 * USD, 25.63 * USD);
  step(20.33 * USD, 25.53 * USD);
  step(20.46 * USD, 25.72 * USD);

  assert(near(filter.x().at<0>(), 1.25398 * mp_units::one, 0.001) &&
         near(filter.x().at<1>(), 0.0627996 * USD, 0.001) &&
         "The state estimates expected at 0.1% accuracy.");

  step(20.32 * USD, 25.74 * USD);
  step(20.05 * USD, 25.33 * USD);
  step(19.84 * USD, 25.04 * USD);
  step(20.2 * USD, 25.67 * USD);
  step(20.22 * USD, 25.71 * USD);
  step(19.93 * USD, 25.4 * USD);
  step(20.04 * USD, 25.61 * USD);
  step(19.94 * USD, 25.82 * USD);
  step(19.73 * USD, 25.46 * USD);
  step(19.64 * USD, 25.29 * USD);
  step(19.21 * USD, 24.56 * USD);
  step(19.19 * USD, 24.6 * USD);
  step(19.31 * USD, 24.75 * USD);
  step(19.24 * USD, 24.99 * USD);
  step(19.36 * USD, 25.22 * USD);
  step(19.12 * USD, 25.06 * USD);
  step(19.1 * USD, 24.99 * USD);
  step(18.81 * USD, 24.77 * USD);
  step(18.86 * USD, 24.6 * USD);
  step(18.56 * USD, 24.05 * USD);
  step(18.73 * USD, 24.25 * USD);
  step(18.32 * USD, 23.46 * USD);
  step(18.14 * USD, 23.39 * USD);
  step(18.16 * USD, 23.58 * USD);
  step(18.05 * USD, 23.29 * USD);
  step(17.8 * USD, 22.86 * USD);
  step(17.79 * USD, 22.66 * USD);
  step(18.0 * USD, 23.34 * USD);
  step(18.23 * USD, 23.41 * USD);
  step(18.34 * USD, 23.99 * USD);
  step(18.6 * USD, 24.29 * USD);
  step(18.75 * USD, 24.6 * USD);
  step(18.47 * USD, 24.34 * USD);
  step(18.53 * USD, 24.29 * USD);
  step(18.37 * USD, 24.07 * USD);
  step(18.63 * USD, 24.55 * USD);
  step(18.83 * USD, 24.9 * USD);
  step(18.91 * USD, 25.29 * USD);
  step(18.72 * USD, 24.97 * USD);
  step(19.13 * USD, 25.59 * USD);
  step(19.19 * USD, 25.77 * USD);
  step(19.33 * USD, 26.08 * USD);
  step(19.02 * USD, 25.73 * USD);
  step(19.17 * USD, 25.91 * USD);
  step(18.88 * USD, 25.57 * USD);
  step(18.97 * USD, 25.69 * USD);
  step(19.07 * USD, 25.96 * USD);
  step(18.7 * USD, 25.54 * USD);
  step(18.47 * USD, 25.31 * USD);
  step(18.46 * USD, 25.58 * USD);

  // The estimated hedge ratio follows the drift of the simulated one, 1.371 on
  // the last day, with a slowly adapting intercept.
  assert(near(filter.x().at<0>(), 1.38158 * mp_units::one, 0.001) &&
         near(filter.x().at<1>(), 0.0680987 * USD, 0.001) &&
         "The state estimates expected at 0.1% accuracy.");
  assert(near(filter.p().at<0, 0>(), 2.00937e-05 * mp_units::one, 0.001) &&
         near(filter.p().at<0, 1>(), -0.000318419 * USD, 0.001) &&
         near(filter.p().at<1, 0>(), -0.000318419 * USD, 0.001) &&
         near(filter.p().at<1, 1>(), 0.00588099 * USD2, 0.001) &&
         "The estimate uncertainty expected at 0.1% accuracy.");

  // On the last day, the spread, the innovation e = 0.286$, rises above its
  // standard deviation √S = 0.190$: the strategy enters a short position on
  // the spread, selling the dependent stock and buying β shares of the
  // independent stock.
  const price spread{filter.y().at<>()};
  const price deviation{sqrt(filter.s())};

  assert(near(spread, 0.286032 * USD, 0.001) &&
         near(deviation, 0.190136 * USD, 0.001) && deviation < spread &&
         "The spread signals a short entry.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
