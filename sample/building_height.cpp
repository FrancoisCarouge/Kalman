#include "fcarouge/kalman.hpp"

#include <cassert>

namespace fcarouge::sample
{
namespace
{
//! @test Estimating the Height of a Building
//!
//! @copyright This example is transcribed from KalmanFilter.NET copyright Alex
//! Becker.
//!
//! @see https://www.kalmanfilter.net/kalman1d.html#ex5
//!
//! @details Assume that we would like to estimate the height of a building
//! using a very imprecise altimeter. We know for sure, that the building height
//! doesn’t change over time, at least during the short measurement process. The
//! true building height is 50 meters. The altimeter measurement error (standard
//! deviation) is 5 meters. The set of ten measurements is: 48.54m, 47.11m,
//! 55.01m, 55.15m, 49.89m, 40.85m, 46.72m, 50.05m, 51.27m, 49.95m.
//!
//! @example building_height.cpp
[[maybe_unused]] constexpr auto building_height{ []() {
  // One-dimensional filter, constant system dynamic model.
  using kalman = kalman<float>;
  kalman k;

  // Initialization
  // One can estimate the building height simply by looking at it. The estimated
  // building height is: 60 meters.
  k.state_x = kalman::state{ 60 };

  // Now we shall initialize the estimate uncertainty. A human’s estimation
  // error (standard deviation) is about 15 meters: σ = 15. Consequently the
  // variance is 225: σ^2 = 225.
  k.estimate_uncertainty_p = kalman::estimate_uncertainty{ 15 * 15 };

  // Prediction
  // Now, we shall predict the next state based on the initialization values.
  assert(60 == k.state_x &&
         "Since our system's dynamic model is constant, i.e. the building "
         "doesn't change its height: 60 meters.");
  assert(225 == k.estimate_uncertainty_p &&
         "The extrapolated estimate uncertainty (variance) also doesn't "
         "change: 225");

  // Measure and Update
  // The first measurement is: z1 = 48.54m. Since the standard deviation σ of
  // the altimeter measurement error is 5, the variance σ^2 would be 25, thus
  // the measurement uncertainty is: r1 = 25.
  k.noise_observation_r = []() {
    return kalman::observation_noise_uncertainty{ 5 * 5 };
  };
  k.update(48.54);

  // And so on.
  k.update(47.11);
  k.update(55.01);
  k.update(55.15);
  k.update(49.89);
  k.update(40.85);
  k.update(46.72);
  k.update(50.05);
  k.update(51.27);
  k.update(49.95);

  // After 10 measurements the filter estimates the height of the building
  // at 49.57m.
  assert(49.57 - 0.001 < k.state_x && k.state_x < 49.57 + 0.001 &&
         "After 10 measurement and update iterations, the building estimated "
         "height is: 49.57m.");

  return 0;
}() };

} // namespace
} // namespace fcarouge::sample
