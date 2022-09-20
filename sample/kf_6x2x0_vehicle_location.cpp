#include "fcarouge/eigen/kalman.hpp"

#include <cassert>
#include <cmath>

namespace fcarouge::eigen::sample {
namespace {
//! @brief Estimating the Vehicle Location
//!
//! @copyright This example is transcribed from KalmanFilter.NET copyright Alex
//! Becker.
//!
//! @see https://www.kalmanfilter.net/multiExamples.html#ex9
//!
//! @details In this example, we would like to estimate the location of the
//! vehicle in the XY plane. The vehicle has an onboard location sensor that
//! reports X and Y coordinates of the system. We assume constant acceleration
//! dynamics. In this example we don't have a control variable u since we don't
//! have control input. Let us assume a vehicle moving in a straight line in the
//! X direction with a constant velocity. After traveling 400 meters the vehicle
//! turns right, with a turning radius of 300 meters. During the turning
//! maneuver, the vehicle experiences acceleration due to the circular motion
//! (an angular acceleration). The measurements period: Δt = 1s (constant). The
//! random acceleration standard deviation: σa = 0.2 m.s^-2.
//!
//! @example kf_6x2x0_vehicle_location.cpp
[[maybe_unused]] auto kf_6x2x0_vehicle_location{[] {
  // A 6x2x0 filter, constant acceleration dynamic model, no control.
  using kalman = kalman<vector<double, 6>, vector<double, 2>>;

  kalman filter;

  // Initialization
  // The state is chosen to be the position, velocity, acceleration in the XY
  // plane: [px, vx, ax, py, vy, ay]. We don't know the vehicle location; we
  // will set initial position, velocity and acceleration to 0.
  filter.x(0., 0., 0., 0., 0., 0.);

  // Since our initial state vector is a guess, we will set a very high estimate
  // uncertainty. The high estimate uncertainty results in a high Kalman Gain,
  // giving a high weight to the measurement.
  filter.p(kalman::estimate_uncertainty{{500, 0, 0, 0, 0, 0},
                                        {0, 500, 0, 0, 0, 0},
                                        {0, 0, 500, 0, 0, 0},
                                        {0, 0, 0, 500, 0, 0},
                                        {0, 0, 0, 0, 500, 0},
                                        {0, 0, 0, 0, 0, 500}});

  // Prediction
  // The process noise matrix Q would be:
  kalman::process_uncertainty q{
      {0.25, 0.5, 0.5, 0, 0, 0}, {0.5, 1, 1, 0, 0, 0}, {0.5, 1, 1, 0, 0, 0},
      {0, 0, 0, 0.25, 0.5, 0.5}, {0, 0, 0, 0.5, 1, 1}, {0, 0, 0, 0.5, 1, 1}};
  q *= 0.2 * 0.2;
  filter.q(std::move(q));

  // The state transition matrix F would be:
  filter.f(kalman::state_transition{{1, 1, 0.5, 0, 0, 0},
                                    {0, 1, 1, 0, 0, 0},
                                    {0, 0, 1, 0, 0, 0},
                                    {0, 0, 0, 1, 1, 0.5},
                                    {0, 0, 0, 0, 1, 1},
                                    {0, 0, 0, 0, 0, 1}});

  // Now we can predict the next state based on the initialization values.
  filter.predict();

  // Measure and Update
  // The dimension of zn is 2x1 and the dimension of xn is 6x1. Therefore the
  // dimension of the observation matrix H shall be 2x6.
  filter.h(kalman::output_model{{1, 0, 0, 0, 0, 0}, {0, 0, 0, 1, 0, 0}});

  // Assume that the x and y measurements are uncorrelated, i.e. error in the x
  // coordinate measurement doesn't depend on the error in the y coordinate
  // measurement. In real-life applications, the measurement uncertainty can
  // differ between measurements. In many systems the measurement uncertainty
  // depends on the measurement SNR (signal-to-noise ratio), angle between
  // sensor (or sensors) and target, signal frequency and many other parameters.
  // For the sake of the example simplicity, we will assume a constant
  // measurement uncertainty: R1 = R2...Rn-1 = Rn = R The measurement error
  // standard deviation: σxm = σym = 3m. The variance 9.
  filter.r(kalman::output_uncertainty{{9, 0}, {0, 9}});

  // The measurement values: z1 = [-393.66, 300.4]
  filter.update(-393.66, 300.4);
  filter.predict();

  // And so on, run a step of the filter, predicting and updating, every
  // measurements period: Δt = 1s (constant, built-in).
  const auto step{[&filter](double position_x, double position_y) {
    filter.update(position_x, position_y);
    filter.predict();
  }};

  step(-375.93, 301.78);

  // Verify the example estimated state at 0.1% accuracy.
  assert(std::abs(1 - filter.x()[0] / -277.8) < 0.001 &&
         std::abs(1 - filter.x()[1] / 148.3) < 0.001 &&
         std::abs(1 - filter.x()[2] / 94.5) < 0.001 &&
         std::abs(1 - filter.x()[3] / 249.8) < 0.001 &&
         std::abs(1 - filter.x()[4] / -85.9) < 0.001 &&
         std::abs(1 - filter.x()[5] / -63.6) < 0.001 &&
         "The state estimates expected at 0.1% accuracy.");

  step(-351.04, 295.1);
  step(-328.96, 305.19);
  step(-299.35, 301.06);
  step(-273.36, 302.05);
  step(-245.89, 300);
  step(-222.58, 303.57);
  step(-198.03, 296.33);
  step(-174.17, 297.65);
  step(-146.32, 297.41);
  step(-123.72, 299.61);
  step(-103.47, 299.6);
  step(-78.23, 302.39);
  step(-52.63, 295.04);
  step(-23.34, 300.09);
  step(25.96, 294.72);
  step(49.72, 298.61);
  step(76.94, 294.64);
  step(95.38, 284.88);
  step(119.83, 272.82);
  step(144.01, 264.93);
  step(161.84, 251.46);
  step(180.56, 241.27);
  step(201.42, 222.98);
  step(222.62, 203.73);
  step(239.4, 184.1);
  step(252.51, 166.12);
  step(266.26, 138.71);
  step(271.75, 119.71);
  step(277.4, 100.41);
  step(294.12, 79.76);
  step(301.23, 50.62);
  step(291.8, 32.99);
  step(299.89, 2.14);

  assert(std::abs(1 - filter.x()[0] / 298.5) < 0.006 &&
         std::abs(1 - filter.x()[1] / -1.65) < 0.006 &&
         std::abs(1 - filter.x()[2] / -1.9) < 0.006 &&
         std::abs(1 - filter.x()[3] / -22.5) < 0.006 &&
         std::abs(1 - filter.x()[4] / -26.1) < 0.006 &&
         std::abs(1 - filter.x()[5] / -0.64) < 0.006 &&
         "The state estimates expected at 0.6% accuracy.");
  assert(std::abs(1 - filter.p()(0, 0) / 11.25) < 0.001 &&
         std::abs(1 - filter.p()(0, 1) / 4.5) < 0.001 &&
         std::abs(1 - filter.p()(0, 2) / 0.9) < 0.001 &&
         std::abs(1 - filter.p()(1, 1) / 2.4) < 0.001 &&
         std::abs(1 - filter.p()(2, 2) / 0.2) < 0.001 &&
         std::abs(1 - filter.p()(3, 3) / 11.25) < 0.001 &&
         "The estimate uncertainty expected at 0.1% accuracy."
         "At this point, the position uncertainty px = py = 5, which means "
         "that the standard deviation of the prediction is square root of 5m.");

  // As you can see, the Kalman Filter tracks the vehicle quite well. However,
  // when the vehicle starts the turning maneuver, the estimates are not so
  // accurate. After a while, the Kalman Filter accuracy improves. While the
  // vehicle travels along the straight line, the acceleration is constant and
  // equal to zero. However, during the turn maneuver, the vehicle experiences
  // acceleration due to the circular motion - the angular acceleration.
  // Although the angular acceleration is constant, the angular acceleration
  // projection on the x and y axes is not constant, therefore ax and ay are not
  // constant. Our Kalman Filter is designed for a constant acceleration model.
  // Nevertheless, it succeeds in tracking maneuvering vehicle due to a properly
  // chosen σa parameter. I would like to encourage the readers to implement
  // this example in software and see how different values of σa of R influence
  // the actual Kalman Filter accuracy, Kalman Gain convergence, and estimation
  // uncertainty.

  return 0;
}()};

} // namespace
} // namespace fcarouge::eigen::sample
