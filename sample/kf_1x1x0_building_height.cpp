#include "fcarouge/kalman.hpp"

#include <cassert>
#include <cmath>

namespace fcarouge::sample {
namespace {
//! @brief Estimating the height of a building.
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
//! @image html ./sample/image/kf_1x1x0_building_height.svg
//!
//! @example kf_1x1x0_building_height.cpp
[[maybe_unused]] auto sample{[] {
  // A one-dimensional filter, constant system dynamic model.
  kalman filter{// One can estimate the building height simply by looking at it.
                // The estimated state building height is: X = 60 meters.
                state{60.},
                // The building height measurement Z.
                output<double>,
                // A human’s estimation error (standard deviation) is about 15
                // meters: σ = 15. Consequently the variance is σ^2 = 225. The
                // estimate uncertainty is: P = 225.
                estimate_uncertainty{225.},
                // Since the standard deviation σ of the altimeter measurement
                // error is 5, the variance σ^2 would be 25, thus the
                // measurement, output uncertainty is: R = 25.
                output_uncertainty{25.}};

  assert(60 == filter.x() &&
         "Since our system's dynamic model is constant, i.e. the building "
         "doesn't change its height: 60 meters.");
  assert(225 == filter.p() &&
         "The extrapolated estimate uncertainty (variance) also doesn't "
         "change: 225");

  // Now, we shall predict the next state based on the initialization values.
  // Note: The prediction operation needs not be performed since the process
  // noise covariance Q is null in this example.
  // Measure and update: the first measurement is: z1 = 48.54m.
  filter.update(48.54);

  // And so on.
  filter.update(47.11);
  filter.update(55.01);
  filter.update(55.15);
  filter.update(49.89);
  filter.update(40.85);
  filter.update(46.72);
  filter.update(50.05);
  filter.update(51.27);
  filter.update(49.95);

  // After 10 measurements the filter estimates the height of the building
  // at 49.57m.
  assert(std::abs(1 - filter.x() / 49.57) < 0.001 &&
         "After 10 measurement and update iterations, the building estimated "
         "height is: 49.57m.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
