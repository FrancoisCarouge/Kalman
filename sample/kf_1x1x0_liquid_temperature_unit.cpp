/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.1
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
#include "fcarouge/unit.hpp"

#include <cassert>

namespace fcarouge::sample {
namespace {
//! @brief Estimating the temperature of the liquid in a tank.
//!
//! @copyright This example is transcribed from KalmanFilter.NET copyright Alex
//! Becker.
//! @copyright This example also transcribes Mateusz Pusz's mp-units Kalman
//! filter sample.
//!
//! @see https://www.kalmanfilter.net/kalman1d.html#ex6
//! @see
//! https://github.com/mpusz/mp-units/blob/master/example/kalman_filter/kalman_filter-example_6.cpp
//!
//! @details We would like to estimate the temperature of the liquid in a tank.
//! We assume that at steady state the liquid temperature is constant. However,
//! some fluctuations in the true liquid temperature are possible. We can
//! describe the system dynamics by the following equation: xn = T + wn where: T
//! is the constant temperature wn is a random process noise with variance q.
//! Let us assume a true temperature of 50 degrees Celsius. The measurements are
//! taken every 5 seconds. The true liquid temperature at the measurement points
//! is: 49.979°C, 50.025°C, 50°C, 50.003°C, 49.994°C, 50.002°C, 49.999°C,
//! 50.006°C, 49.998°C, and 49.991°C. The set of measurements is: 49.95°C,
//! 49.967°C, 50.1°C, 50.106°C, 49.992°C, 49.819°C, 49.933°C, 50.007°C,
//! 50.023°C, and 49.99°C.
//!
//! @example kf_1x1x0_liquid_temperature_unit.cpp
[[maybe_unused]] auto sample{[] {
  // A one-dimensional filter, constant system dynamic model.
  kalman filter{
      // We initialize the Kalman filter and predict the next state (which is
      // the first state). We don't know what the temperature of the liquid is,
      // and our guess is: estimated state X = 10°C.
      state{temperature{point<deg_C>(10.)}},
      // The measured liquid temperature Z.
      output<temperature>,
      // Our guess is very imprecise, so we set our initialization estimate
      // error σ to 100. The estimate uncertainty p of the initialization is the
      // error variance σ^2: P = p0,0 = 100^2 = 10,000. This variance is very
      // high. If we initialize with a more meaningful value, we will get faster
      // Kalman filter convergence.
      estimate_uncertainty{delta<deg_C2>(10'000.)},
      // We have an accurate model, thus we set the process uncertainty noise
      // variance Q to 0.0001.
      process_uncertainty{delta<deg_C2>(0.000'1)},
      // Since the measurement error of the thermometer is σ = 0.1, the variance
      // σ^2 would be 0.01, thus the measurement, output uncertainty is: R = r1
      // = 0.01. The measurement error (standard deviation) is 0.1 degrees
      // Celsius.
      output_uncertainty{delta<deg_C2>(0.01)}};

  // Now, we shall predict the next state based on the initialization values.
  filter.predict();

  assert(temperature{point<deg_C>(10.)} == filter.x() &&
         "Since our model has constant dynamics, the predicted estimate is "
         "equal to the current estimate: x^1,0 = 10°C.");
  assert(10'000.000'1 * deg_C2 == filter.p() &&
         "The extrapolated estimate uncertainty (variance): p1,0 = p0,0 + q = "
         "10'000 + 0.000'1 = 10'000.000'1.");

  // The first measurement value: z1 = 49.95°C. Measure and update.
  filter.update(temperature{point<deg_C>(49.95)});

  assert(abs(1 - filter.k() / 0.999'999) < 0.000'1 &&
         "The gain expected at 0.01% accuracy.");

  // And so on, run a step of the filter, predicting and updating, every
  // measurements period: Δt = 5s (constant).
  const auto step{[&filter](temperature output_z) {
    filter.predict();
    filter.update(output_z);
  }};

  step(temperature{point<deg_C>(49.967)});
  step(temperature{point<deg_C>(50.1)});
  step(temperature{point<deg_C>(50.106)});
  step(temperature{point<deg_C>(49.992)});
  step(temperature{point<deg_C>(49.819)});
  step(temperature{point<deg_C>(49.933)});
  step(temperature{point<deg_C>(50.007)});
  step(temperature{point<deg_C>(50.023)});
  step(temperature{point<deg_C>(49.99)});

  // The estimate uncertainty quickly goes down, after 10 measurements:
  assert(0.001'2 * deg_C2 < filter.p() && filter.p() < 0.001'3 * deg_C2 &&
         "The estimate uncertainty expected at 5% accuracy."
         "The estimate uncertainty is 0.0013, i.e. the estimate error standard "
         "deviation is: 0.036°C.");
  assert(temperature{point<deg_C>(49.98)} < filter.x() &&
         filter.x() < temperature{point<deg_C>(49.99)} &&
         "The state estimates expected at 0.1% accuracy."
         "The filter estimates the liquid temperature at 49.988°C.");
  assert(0.126'4 < filter.k() && filter.k() < 0.126'5 &&
         "The gain expected at 0.1% accuracy.");

  // So we can say that the liquid temperature estimate is: 49.988 ± 0.036°C.
  // In this example we've measured a liquid temperature using the
  // one-dimensional Kalman filter. Although the system dynamics include
  // a random process noise, the Kalman filter can provide good estimation.

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
