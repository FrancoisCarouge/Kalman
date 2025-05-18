/*  __          _      __  __          _   _
| |/ /    /\   | |    |  \/  |   /\   | \ | |
| ' /    /  \  | |    | \  / |  /  \  |  \| |
|  <    / /\ \ | |    | |\/| | / /\ \ | . ` |
| . \  / ____ \| |____| |  | |/ ____ \| |\  |
|_|\_\/_/    \_\______|_|  |_/_/    \_\_| \_|

Kalman Filter
Version 0.5.2
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
//! @brief Estimating the height of a building.
//!
//! @copyright This example is transcribed from KalmanFilter.NET copyright Alex
//! Becker.
//! @copyright This example also transcribes Mateusz Pusz's mp-units Kalman
//! filter sample.
//!
//! @see https://www.kalmanfilter.net/kalman1d.html#ex5
//! @see
//! https://github.com/mpusz/mp-units/blob/master/example/kalman_filter/kalman_filter-example_5.cpp
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
//! @example kf_1x1x0_building_height_unit.cpp
[[maybe_unused]] auto sample{[] {
  // A one-dimensional filter, constant system dynamic model.
  kalman filter{// One can estimate the building height simply by looking at it.
                // The estimated state building height is: X = 60 meters.
                state{height{60. * m}},
                // The building height measurement Z in meters.
                output<height>,
                // A human’s estimation error (standard deviation) is about 15
                // meters: σ = 15. Consequently the variance is σ^2 = 225. The
                // estimate uncertainty is: P = 225 m^2.
                estimate_uncertainty{225. * m2},
                // Since the standard deviation σ of the altimeter measurement
                // error is 5, the variance σ^2 would be 25, thus the
                // measurement, output uncertainty is: R = 25 m^2.
                output_uncertainty{25. * m2}};

  assert(height{60. * m} == filter.x() &&
         "Since our system's dynamic model is constant, i.e. the building "
         "doesn't change its height: 60 meters.");

  using uncertainty = kalman_internal::product<height, height>;

  assert(uncertainty{225. * m2} == filter.p() &&
         "The extrapolated estimate uncertainty (variance) also doesn't "
         "change: 225");

  // Now, we shall predict the next state based on the initialization values.
  // Note: The prediction operation needs not be performed since the process
  // noise covariance Q is null in this example.
  // Measure and update: the first measurement is: z1 = 48.54m.
  filter.update(height{48.54 * m});

  // And so on.
  filter.update(height{47.11 * m});
  filter.update(height{55.01 * m});
  filter.update(height{55.15 * m});
  filter.update(height{49.89 * m});
  filter.update(height{40.85 * m});
  filter.update(height{46.72 * m});
  filter.update(height{50.05 * m});
  filter.update(height{51.27 * m});
  filter.update(height{49.95 * m});

  // After 10 measurements the filter estimates the height of the building
  // at 49.57m.
  assert(height{49.569 * m} < filter.x() && filter.x() < height{49.571 * m} &&
         "After 10 measurement and update iterations, the building estimated "
         "height is: 49.57m.");

  //! @todo Add format printing, verification.

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
