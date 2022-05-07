#include "fcarouge/kalman.hpp"

#include <cassert>

namespace fcarouge::sample
{
namespace
{
//! @brief Estimating the Position of a Dog
//!
//! @copyright This example is transcribed from Kalman and Bayesian Filters in
//! Python copyright Roger Labbe
//!
//! @see
//! https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/04-One-Dimensional-Kalman-Filters.ipynb
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
//! @example dog_position1d.cpp
[[maybe_unused]] auto dog_position{ [] {
  using kalman = kalman<double>;
  kalman k;

  // Initialization
  // This is the dog's initial position expressed as a Gaussian. The position is
  // 0 meters, and the variance to 400 m, which is a standard deviation of 20
  // meters. You can think of this as saying "I believe with 99.7% accuracy the
  // position is 0 plus or minus 60 meters". This is because with Gaussians
  // ~99.7% of values fall within of the mean.
  k.x(1.);
  k.p(20 * 20.);

  // Prediction
  // Variance in the dog's movement. The process variance is how much error
  // there is in the process model. Dogs rarely do what we expect, and things
  // like hills or the whiff of a squirrel will change his progress.
  k.q(1.);

  // Measure and Update
  // Variance in the sensor. The meaning of sensor variance should be clear - it
  // is how much variance there is in each measurement.
  k.r(2.);

  // We are predicting that at each time step the dog moves forward one meter.
  // This is the process model - the description of how we think the dog moves.
  // How do I know the velocity? Magic? Consider it a prediction, or perhaps we
  // have a secondary velocity sensor. Please accept this simplification for
  // now.
  k.g(1.);

  k.predict(1.);
  k.update(1.354);
  k.predict(1.);
  k.update(1.882);
  k.predict(1.);
  k.update(4.341);
  k.predict(1.);
  k.update(7.156);
  k.predict(1.);
  k.update(6.939);
  k.predict(1.);
  k.update(6.844);
  k.predict(1.);
  k.update(9.847);
  k.predict(1.);
  k.update(12.553);
  k.predict(1.);
  k.update(16.273);
  k.predict(1.);
  k.update(14.8);

  assert(
      15.053 - 0.001 < k.x() && k.x() < 15.053 + 0.001 &&
      "Here we can see that the variance converges to 2.1623 in 9 steps. This "
      "means that we have become very confident in our position estimate. It "
      "is equal to meters. Contrast this to the sensor's meters.");

  return 0;
}() };

} // namespace
} // namespace fcarouge::sample
