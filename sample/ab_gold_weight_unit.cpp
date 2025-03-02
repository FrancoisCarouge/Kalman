#include "fcarouge/kalman.hpp"
#include "fcarouge/unit.hpp"

#include <cassert>
#include <cmath>

namespace fcarouge::sample {
namespace {
//! @brief Estimating the weight of gold.
//!
//! @copyright This example is transcribed from KalmanFilter.NET copyright Alex
//! Becker.
//! @copyright This example also transcribes Mateusz Pusz's mp-units Kalman
//! filter sample.
//!
//! @see https://www.kalmanfilter.net/alphabeta.html#ex1
//!
//! @details Estimate the state of a static system. Estimate the weight of a
//! gold bar. The measurements include random noise. The system is the gold bar,
//! and the system state is the weight of the gold bar. The dynamic model of the
//! system is constant since we assume that the weight doesn't change over short
//! periods. We can make multiple measurements and average them.
//!
//! @example ab_gold_weight_unit.cpp
[[maybe_unused]] auto sample{[] {
  // A one-dimensional filter, constant system dynamic model.
  kalman filter{
      state{1000.}, output<double>
      // Our initial guess of the gold bar weight is 1000 grams. The initial
      // guess is used only once for the filter initiation. Thus, it won't be
      // required for successive iterations.
  };

  // The weight of the gold bar is not supposed to change. Therefore, the
  // dynamic model of the system is static. Our next state estimate (prediction)
  // equals the initialization:
  filter.predict();
  // filter.x() == 1000

  // Making the weight measurement with the scales:
  // filter.update(996)
  filter.predict();
  // filter.x() == 996

  filter.update(994);
  filter.update(1021);
  filter.update(1000);
  filter.update(1002);
  filter.update(1010);
  filter.update(983);
  filter.update(971);
  filter.update(993);
  filter.update(1023);
  // --> 999.3g

  // We can stop here. The gain decreases with each measurement. Therefore, the
  // contribution of each successive measurement is lower than the contribution
  // of the previous measurement. We get pretty close to the true weight, which
  // is 1000g. If we were making more measurements, we would get closer to the
  // true value.

  return 0;
}()};
} // namespace
} // namespace fcarouge::sample
