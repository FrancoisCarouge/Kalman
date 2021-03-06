#include "fcarouge/kalman.hpp"

#include <cassert>

namespace fcarouge::sample
{
namespace
{
//! @brief Estimating the Temperature of the Liquid in a Tank
//!
//! @copyright This example is transcribed from KalmanFilter.NET copyright Alex
//! Becker.
//!
//! @see https://www.kalmanfilter.net/kalman1d.html#ex6
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
//! @example liquid_temperature.cpp
[[maybe_unused]] auto liquid_temperature{ [] {
  // A one-dimensional filter, constant system dynamic model.
  kalman k;

  // Initialization
  // Before the first iteration, we must initialize the Kalman filter and
  // predict the next state (which is the first state). We don't know what the
  // temperature of the liquid is, and our guess is 10°C.
  k.x(10.);

  // Our guess is very imprecise, so we set our initialization estimate error σ
  // to 100. The estimate uncertainty of the initialization is the error
  // variance σ^2: p0,0 = 100^2 = 10,000. This variance is very high. If we
  // initialize with a more meaningful value, we will get faster Kalman filter
  // convergence.
  k.p(100 * 100.);

  // Prediction
  // Now, we shall predict the next state based on the initialization values. We
  // think that we have an accurate model, thus we set the process noise
  // variance q to 0.0001.
  k.q(0.0001);

  k.predict();

  assert(10 == k.x() &&
         "Since our model has constant dynamics, the predicted estimate is "
         "equal to the current estimate: x^1,0 = 10°C.");
  assert(10000.0001 - 0.001 < k.p() && k.p() < 10000.0001 + 0.001 &&
         "The extrapolated estimate uncertainty (variance): p1,0 = p0,0 + q = "
         "10000 + 0.0001 = 10000.0001.");

  // Measure and Update
  // The measurement value: z1 = 49.95°C. Since the measurement error is σ =
  // 0.1, the variance σ^2 would be 0.01, thus the measurement uncertainty is:
  // r1 = 0.01. The measurement error (standard deviation) is 0.1 degrees
  // Celsius.
  k.r(0.1 * 0.1);

  k.update(49.95);

  assert(0.999999 - 0.001 < k.k() && k.k() < 0.999999 + 0.001);

  // And so on, run a step of the filter, predicting and updating, every
  // measurements period: Δt = 5s (constant).
  k(49.967);
  k(50.1);
  k(50.106);
  k(49.992);
  k(49.819);
  k(49.933);
  k(50.007);
  k(50.023);
  k(49.99);

  // The estimate uncertainty quickly goes down, after 10 measurements:
  assert(0.0013 - 0.0001 < k.p() && k.p() < 0.0013 + 0.0001 &&
         "The estimate uncertainty is 0.0013, i.e. the estimate error standard "
         "deviation is: 0.036°C.");
  assert(49.988 - 0.0001 < k.x() && k.x() < 49.988 + 0.0001 &&
         "The filter estimates the liquid temperature at 49.988°C.");
  assert(0.1265 - 0.001 < k.k() && k.k() < 0.1265 + 0.001);
  // So we can say that the liquid temperature estimate is: 49.988 ± 0.036°C.
  // In this example we've measured a liquid temperature using the
  // one-dimensional Kalman filter. Although the system dynamics include a
  // random process noise, the Kalman filter can provide good estimation.

  return 0;
}() };

} // namespace
} // namespace fcarouge::sample
