#include "fcarouge/eigen/kalman.hpp"

#include <cassert>
#include <iostream>

namespace fcarouge::eigen::sample {
namespace {
//! @brief Estimating the Position of a Dog
//!
//! @copyright This example is transcribed from Kalman and Bayesian Filters in
//! Python copyright Roger Labbe
//!
//! @see
//! https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/06-Multivariate-Kalman-Filters.ipynb
//!
//! @details Assume that in our latest hackathon someone created an RFID tracker
//! that provides a reasonably accurate position of the dog. The sensor returns
//! the distance of the dog from the left end of the hallway in meters. So, 23.4
//! would mean the dog is 23.4 meters from the left end of the hallway. The
//! sensor is not perfect. A reading of 23.4 could correspond to the dog being
//! at 23.7, or 23.0. However, it is very unlikely to correspond to a position
//! of 47.6. Testing during the hackathon confirmed this result - the sensor is
//! 'reasonably' accurate, and while it had errors, the errors are small.
//! Furthermore, the errors seemed to be evenly distributed on both sides of the
//! true position; a position of 23 m would equally likely be measured as 22.9
//! or 23.1. Perhaps we can model this with a Gaussian. We predict that the dog
//! is moving. This prediction is not perfect. Sometimes our prediction will
//! overshoot, sometimes it will undershoot. We are more likely to undershoot or
//! overshoot by a little than a lot. Perhaps we can also model this with a
//! Gaussian.
//!
//! @example dog_position2d.cpp
[[maybe_unused]] auto dog_position2d{[] {
  using kalman = kalman<vector<double, 2>, double, double>;

  kalman k;

  // Initialization
  k.x(10., 4.5);
  k.p(kalman::estimate_uncertainty{{500., 0.}, {0., 49.}});

  // Prediction
  const double dt{0.1}; // seconds
  k.f(kalman::state_transition{{1, dt}, {0, 1}});

  k.predict();
  k.predict();
  k.predict();
  k.predict();
  k.predict();

  assert(12.25 - 0.001 < k.x()(0) && k.x()(0) < 12.25 + 0.001 &&
         4.5 - 0.001 < k.x()(1) && k.x()(1) < 4.5 + 0.001);
  assert(512.25 - 0.001 < k.p()(0) && k.p()(0) < 512.25 + 0.001 &&
         24.5 - 0.001 < k.p()(1) && k.p()(1) < 24.5 + 0.001 &&
         24.5 - 0.001 < k.p()(2) && k.p()(2) < 24.5 + 0.001 &&
         49 - 0.001 < k.p()(3) && k.p()(3) < 49 + 0.001);

  k.q(kalman::process_uncertainty{{0.588, 1.175}, {1.175, 2.35}});
  k.g(kalman::input_control{0., 0.});
  k.predict(0.);

  assert(12.7 - 0.001 < k.x()(0) && k.x()(0) < 12.7 + 0.001 &&
         4.5 - 0.001 < k.x()(1) && k.x()(1) < 4.5 + 0.001);

  //   std::cout << k.p() << std::endl << std::endl;

  // assert(680.587 - 0.001 < k.p()(0) && k.p()(0) < 680.587 + 0.001 &&
  //        301.175 - 0.001 < k.p()(1) && k.p()(1) < 301.175 + 0.001 &&
  //        301.175 - 0.001 < k.p()(2) && k.p()(2) < 301.175 + 0.001 &&
  //        502.35 - 0.001 < k.p()(3) && k.p()(3) < 502.35 + 0.001);

  // Measure and Update
  k.h(kalman::output_model{1., 0.});
  k.r(kalman::output_uncertainty{5.});

  k.update(1.);

  //   std::cout << k.x() << std::endl << std::endl;

  // assert(
  //     15.053 - 0.001 < k.x() && k.x() < 15.053 + 0.001 &&
  //     "Here we can see that the variance converges to 2.1623 in 9 steps. This
  //     " "means that we have become very confident in our position estimate.
  //     It " "is equal to meters. Contrast this to the sensor's meters.");

  return 0;
}()};

} // namespace
} // namespace fcarouge::eigen::sample
