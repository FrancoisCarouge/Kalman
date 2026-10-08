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

namespace fcarouge::sample {
namespace {
template <typename... Types> using vector = column_vector<double, Types...>;
using price_slope = mp_units::quantity<USD / d>;
using state_t = vector<price, price_slope>;
using output_t = vector<price>;

//! @brief Forecasting the short-term stock price.
//!
//! @copyright This example implements the local linear trend model as
//! presented by Andrew C. Harvey and by James Durbin and Siem Jan Koopman. The
//! closing prices are simulated and the expected values computed independently.
//!
//! @see Andrew C. Harvey, Forecasting, Structural Time Series Models and the
//! Kalman Filter, Cambridge University Press, 1990, chapter 3 State space
//! models and the Kalman filter, https://doi.org/10.1017/CBO9781107049994
//! @see James Durbin and Siem Jan Koopman, Time Series Analysis by State Space
//! Methods, Second Edition, Oxford University Press, 2012, chapter 3 Linear
//! state space models,
//! https://doi.org/10.1093/acprof:oso/9780199641178.001.0001
//!
//! @details The filter forecasts the next daily closing price of a stock. The
//! closing price y is an unobserved level μ plus a transient noise ε, such as
//! the bid-ask bounce. The level drifts with a slope ν:
//! - yt = μt + εt, εt ~ N(0, σε^2).
//! - μt+1 = μt + νt + ξt, ξt ~ N(0, σξ^2).
//! - νt+1 = νt + ζt, ζt ~ N(0, σζ^2).
//! The predicted level is the forecast, with its uncertainty. Forty daily
//! closing prices are simulated from the model with σε = 1$, σξ = 0.5$, and
//! σζ = 0.1$ per day. Not trading advice.
//!
//! @image html ./sample/image/kf_2x1x0_stock_price_unit.svg
//!
//! @example kf_2x1x0_stock_price_unit.cpp
[[maybe_unused]] const auto sample{[] -> int {
  // A 2x1x0 filter, local linear trend dynamic model, no control, constant
  // daily time step.
  kalman filter{
      // The state X is chosen to be the price level and slope: [μ, ν]. We don't
      // know the stock price; we will set the initial level and slope to 0.
      state{state_t{price{0. * USD}, price_slope{0. * USD / d}}},
      // The filter observes the output Z closing price [$].
      output<output_t>,
      // Since our initial state is a guess, we will set a very high estimate
      // uncertainty, an approximation of the diffuse initialization of the
      // model. The first closing price then sets the level and the second one
      // sets the slope.
      estimate_uncertainty{[]() -> auto {
        using estimate_uncertainty_t = kf::ᴀʙᵀ<state_t, state_t>;
        estimate_uncertainty_t value;
        value.at<0, 0>(1000. * USD2);
        value.at<1, 1>(1000. * USD2 / d / d);
        return value;
      }()},
      // The process uncertainty Q holds the variances of the level and slope
      // disturbances: σξ^2 = 0.25 $^2 and σζ^2 = 0.01 $^2/d^2.
      process_uncertainty{[]() -> auto {
        using process_uncertainty_t = kf::ᴀʙᵀ<state_t, state_t>;
        process_uncertainty_t value;
        value.at<0, 0>(0.25 * USD2);
        value.at<1, 1>(0.01 * USD2 / d / d);
        return value;
      }()},
      // The output uncertainty R is the variance of the transient fluctuations
      // of the closing price around its level: σε^2 = 1 $^2.
      output_uncertainty{1. * USD2},
      // The output model H observes the level only.
      output_model{[]() -> auto {
        using output_model_t = kf::evaluate<kf::quotient<output_t, state_t>>;
        output_model_t value;
        value.at<0>(1.);
        return value;
      }()},
      // The state transition F adds the slope over one day to the level.
      state_transition{[]() -> auto {
        using state_transition_t = kf::evaluate<kf::quotient<state_t, state_t>>;
        state_transition_t value;
        value.at<0, 0>(1.);
        value.at<0, 1>(1. * d);
        value.at<1, 1>(1.);
        return value;
      }()}};

  // Verifies a value at the relative accuracy of its expectation.
  const auto near{
      [](const auto &value, const auto &expected, double accuracy) -> bool {
        return abs(value - expected) < accuracy * abs(expected);
      }};

  // Run a step of the filter, updating with the closing price of the day and
  // predicting the closing price of the next day, every trading day.
  const auto step{[&filter](price close) -> void {
    filter.update(close);
    filter.predict();
  }};

  step(100.87 * USD);

  // The first closing price sets the level, the slope remains unknown.
  assert(near(filter.x().at<0>(), 100.77 * USD, 0.001) &&
         abs(filter.x().at<1>()) < 0.001 * USD / d &&
         "The state estimates expected at 0.1% accuracy.");

  step(100.45 * USD);
  step(99.86 * USD);

  assert(near(filter.x().at<0>(), 99.447 * USD, 0.001) &&
         near(filter.x().at<1>(), -0.4545 * USD / d, 0.001) &&
         "The state estimates expected at 0.1% accuracy.");
  assert(near(filter.p().at<0, 0>(), 2.7337 * USD2, 0.001) &&
         near(filter.p().at<0, 1>(), 1.1372 * USD2 / d, 0.001) &&
         near(filter.p().at<1, 0>(), 1.1372 * USD2 / d, 0.001) &&
         near(filter.p().at<1, 1>(), 0.64686 * USD2 / d / d, 0.001) &&
         "The estimate uncertainty expected at 0.1% accuracy.");

  step(99.32 * USD);
  step(100.53 * USD);
  step(101.14 * USD);
  step(98.63 * USD);
  step(99.31 * USD);
  step(103.04 * USD);
  step(96.89 * USD);
  step(99.94 * USD);
  step(102.41 * USD);
  step(101.85 * USD);
  step(101.11 * USD);
  step(100.59 * USD);
  step(102.15 * USD);
  step(99.28 * USD);
  step(102.62 * USD);
  step(101.88 * USD);
  step(102.56 * USD);
  step(103.11 * USD);
  step(101.83 * USD);
  step(103.25 * USD);
  step(101.97 * USD);
  step(101.6 * USD);
  step(101.6 * USD);
  step(100.14 * USD);
  step(102.23 * USD);
  step(100.59 * USD);
  step(100.6 * USD);
  step(98.47 * USD);
  step(98.38 * USD);
  step(99.53 * USD);
  step(98.56 * USD);
  step(96.41 * USD);
  step(95.41 * USD);
  step(94.07 * USD);
  step(94.43 * USD);
  step(91.81 * USD);
  step(91.91 * USD);

  // The price turned down: the filter estimates a negative trend and forecasts
  // the next closing price below the last one, at 91.17$. The forecast
  // uncertainty of the closing price is the level uncertainty plus the
  // transient fluctuations: σ = √(0.948 + 1) ≈ 1.40$.
  assert(near(filter.x().at<0>(), 91.172 * USD, 0.001) &&
         near(filter.x().at<1>(), -0.78336 * USD / d, 0.001) &&
         "The state estimates expected at 0.1% accuracy.");
  assert(near(filter.p().at<0, 0>(), 0.94782 * USD2, 0.001) &&
         near(filter.p().at<0, 1>(), 0.13956 * USD2 / d, 0.001) &&
         near(filter.p().at<1, 0>(), 0.13956 * USD2 / d, 0.001) &&
         near(filter.p().at<1, 1>(), 0.077913 * USD2 / d / d, 0.001) &&
         "The estimate uncertainty expected at 0.1% accuracy.");
  assert(near(filter.k().at<0>(), 0.48661 * mp_units::one, 0.001) &&
         near(filter.k().at<1>(), 0.071652 / d, 0.001) &&
         "The steady-state gain weighs the closing price about as much as the "
         "forecast.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
