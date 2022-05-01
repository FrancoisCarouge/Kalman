#include "fcarouge/kalman_eigen.hpp"

#include <cassert>

namespace fcarouge::sample
{
namespace
{
//! @test Estimating the Vehicule Location
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
[[maybe_unused]] auto vehicule_location{ [] {
  // A 6x2x0 filter, constant acceleration dynamic model, no control.
  using kalman = eigen::kalman<double, 6, 2, 0>;
  kalman k;

  // Initialization
  // The state is chosen to be the position, velocity, acceleration in the XY
  // plane: [px, vx, ax, py, vy, ay]. We don't know the vehicle location; we
  // will set initial position, velocity and acceleration to 0.
  k.x(0., 0., 0., 0., 0., 0.);

  // Since our initial state vector is a guess, we will set a very high estimate
  // uncertainty. The high estimate uncertainty results in a high Kalman Gain,
  // giving a high weight to the measurement.
  k.p(kalman::estimate_uncertainty{ { 500, 0, 0, 0, 0, 0 },
                                    { 0, 500, 0, 0, 0, 0 },
                                    { 0, 0, 500, 0, 0, 0 },
                                    { 0, 0, 0, 500, 0, 0 },
                                    { 0, 0, 0, 0, 500, 0 },
                                    { 0, 0, 0, 0, 0, 500 } });

  // Prediction
  // The process noise matrix Q would be:
  k.q(kalman::process_uncertainty{ { 1, 0.5, 0.5, 0, 0, 0 },
                                   { 0.5, 1, 1, 0, 0, 0 },
                                   { 0.5, 1, 1, 0, 0, 0 },
                                   { 0, 0, 0, 0.25, 0.5, 0.5 },
                                   { 0, 0, 0, 0.5, 1, 1 },
                                   { 0, 0, 0, 0.5, 1, 1 } });

  // The state transition matrix F would be:
  k.f(kalman::state_transition{ { 1, 1, 0.5, 0, 0, 0 },
                                { 0, 1, 1, 0, 0, 0 },
                                { 0, 0, 1, 0, 0, 0 },
                                { 0, 0, 0, 1, 1, 0.5 },
                                { 0, 0, 0, 0, 1, 1 },
                                { 0, 0, 0, 0, 0, 1 } });

  // Now we can predict the next state based on the initialization values.
  k.predict();

  // Measure and Update
  // The dimension of zn is 2x1 and the dimension of xn is 6x1. Therefore the
  // dimension of the observation matrix H shall be 2x6.
  k.h(kalman::output_model{ { 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 1, 0, 0 } });

  // Assume that the x and y measurements are uncorrelated, i.e. error in the x
  // coordinate measurement doesn't depend on the error in the y coordinate
  // measurement. In real-life applications, the measurement uncertainty can
  // differ between measurements. In many systems the measurement uncertainty
  // depends on the measurement SNR (signal-to-noise ratio), angle between
  // sensor (or sensors) and target, signal frequency and many other parameters.
  // For the sake of the example simplicity, we will assume a constant
  // measurement uncertainty: R1 = R2...Rn-1 = Rn = R The measurement error
  // standard deviation: σxm = σym = 3m. The variance 9.
  k.r(kalman::output_uncertainty{ { 9, 0 }, { 0, 9 } });

  // The measurement values: z1 = [-393.66, 300.4]
  k.observe(-393.66, 300.4);

  // And so on, every measurements period: Δt = 1s (constant, built-in).
  k.predict();
  k.observe(-375.93, 301.78);
  k.predict();
  k.observe(-351.04, 295.1);
  k.predict();
  k.observe(-328.96, 305.19);
  k.predict();
  k.observe(-299.35, 301.06);
  k.predict();
  k.observe(-273.36, 302.05);
  k.predict();
  k.observe(-245.89, 300);
  k.predict();
  k.observe(-222.58, 303.57);
  k.predict();
  k.observe(-198.03, 296.33);
  k.predict();
  k.observe(-174.17, 297.65);
  k.predict();
  k.observe(-146.32, 297.41);
  k.predict();
  k.observe(-123.72, 299.61);
  k.predict();
  k.observe(-103.47, 299.6);
  k.predict();
  k.observe(-78.23, 302.39);
  k.predict();
  k.observe(-52.63, 295.04);
  k.predict();
  k.observe(-23.34, 300.09);
  k.predict();
  k.observe(25.96, 294.72);
  k.predict();
  k.observe(49.72, 298.61);
  k.predict();
  k.observe(76.94, 294.64);
  k.predict();
  k.observe(95.38, 284.88);
  k.predict();
  k.observe(119.83, 272.82);
  k.predict();
  k.observe(144.01, 264.93);
  k.predict();
  k.observe(161.84, 251.46);
  k.predict();
  k.observe(180.56, 241.27);
  k.predict();
  k.observe(201.42, 222.98);
  k.predict();
  k.observe(222.62, 203.73);
  k.predict();
  k.observe(239.4, 184.1);
  k.predict();
  k.observe(252.51, 166.12);
  k.predict();
  k.observe(266.26, 138.71);
  k.predict();
  k.observe(271.75, 119.71);
  k.predict();
  k.observe(277.4, 100.41);
  k.predict();
  k.observe(294.12, 79.76);
  k.predict();
  k.observe(301.23, 50.62);
  k.predict();
  k.observe(291.8, 32.99);
  k.predict();
  k.observe(299.89, 2.14);

  assert(5 - 0.0001 < k.p()(0, 0) && k.p()(0, 0) < 5 + 1.84 &&
         5 - 0.0001 < k.p()(3, 3) && k.p()(3, 3) < 5 + 1.751 &&
         "At this point, the position uncertainty px = py = 5, which means "
         "that the standard deviation of the prediction is square root of 5m.");

  k.predict();

  assert(298.5 - 1.7 < k.x()(0) && k.x()(0) < 298.5 + 0.0001 &&
         -1.65 - 1 < k.x()(1) && k.x()(1) < -1.65 + 0.0001 &&
         -1.9 - 0.3 < k.x()(2) && k.x()(2) < -1.9 + 0.0001 &&
         -22.5 - 0.5 < k.x()(3) && k.x()(3) < -22.5 + 0.0001 &&
         -26.1 - 1.5 < k.x()(4) && k.x()(4) < -26.1 + 0.0001 &&
         -0.64 - 0.7 < k.x()(5) && k.x()(5) < -0.64 + 0.0001);

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
}() };

} // namespace
} // namespace fcarouge::sample
