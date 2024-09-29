#include "fcarouge/kalman.hpp"

#include <cassert>
#include <cmath>
import mp_units;

namespace fcarouge::test {
namespace {
//! @test Verifies compatibility with
[[maybe_unused]] auto sample{[] {
  kalman filter{state{60.}, output<double>, estimate_uncertainty{225.},
                output_uncertainty{25.}};

  assert(60 == filter.x() &&
         "Since our system's dynamic model is constant, i.e. the building "
         "doesn't change its height: 60 meters.");
  assert(225 == filter.p() &&
         "The extrapolated estimate uncertainty (variance) also doesn't "
         "change: 225");

  filter.update(48.54);

  filter.update(47.11);
  filter.update(55.01);
  filter.update(55.15);
  filter.update(49.89);
  filter.update(40.85);
  filter.update(46.72);
  filter.update(50.05);
  filter.update(51.27);
  filter.update(49.95);

  assert(std::abs(1 - filter.x() / 49.57) < 0.001 &&
         "After 10 measurement and update iterations, the building estimated "
         "height is: 49.57m.");

  return 0;
}()};
} // namespace
} // namespace fcarouge::test
