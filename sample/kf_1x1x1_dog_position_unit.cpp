/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.3
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
//! @brief Estimating the position of a dog.
//!
//! @copyright This example is transcribed from Kalman and Bayesian Filters in
//! Python copyright Roger Labbe.
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
//! @example kf_1x1x1_dog_position_unit.cpp
[[maybe_unused]] auto sample{[] {
  kalman filter{
      // This is the dog's initial position expressed as a Gaussian. The state X
      // position is 0 meters.
      state{position{1. * m}},
      // The measured output position Z.
      output<position>,
      // We are predicting that at each time step the dog moves forward one
      // meter. This is the input U process model - the description of how we
      // think the dog moves. How do I know the velocity? Magic? Consider it a
      // prediction, or perhaps we have a secondary velocity sensor. Please
      // accept this simplification for now.
      input<position>,
      // The variance to 400 m, which is a standard deviation of 20 meters. You
      // can think of this as saying "I believe with 99.7% accuracy the position
      // is 0 plus or minus 60 meters". This is because with Gaussians ~99.7% of
      // values fall within of the mean. The estimate uncertainty P:
      estimate_uncertainty{20. * m2},
      // Variance in the dog's movement. The process variance is how much error
      // there is in the process model. Dogs rarely do what we expect, and
      // things like hills or the whiff of a squirrel will change his progress.
      // The process uncertainty Q:
      process_uncertainty{1. * m2},
      // Variance in the sensor. The meaning of sensor variance is how much
      // variance there is in each measurement. The output uncertainty R:
      output_uncertainty{2. * m2}};

  filter.predict(position{1. * m});
  filter.update(position{1.354 * m});
  filter.predict(position{2.352 * m});
  filter.update(position{1.882 * m});
  filter.predict(position{3.070 * m});
  filter.update(position{4.341 * m});
  filter.predict(position{4.736 * m});
  filter.update(position{7.156 * m});
  filter.predict(position{6.960 * m});
  filter.update(position{6.939 * m});
  filter.predict(position{7.949 * m});
  filter.update(position{6.844 * m});
  filter.predict(position{8.396 * m});
  filter.update(position{9.847 * m});
  filter.predict(position{10.122 * m});
  filter.update(position{12.553 * m});
  filter.predict(position{12.338 * m});
  filter.update(position{16.273 * m});
  filter.predict(position{15.305 * m});
  filter.update(position{14.8 * m});

  assert(
      position{15.052 * m} < filter.x() && filter.x() < position{15.054 * m} &&
      "Here we can see that the variance converges to 2.1623 in 9 steps. This "
      "means that we have become very confident in our position estimate. It "
      "is equal to meters. Contrast this to the sensor's meters.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
