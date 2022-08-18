#include "fcarouge/eigen/kalman.hpp"

#include <cassert>
#include <chrono>

namespace fcarouge::eigen::sample
{
namespace
{
//! @brief Estimating the Rocket Altitude
//!
//! @copyright This example is transcribed from KalmanFilter.NET copyright Alex
//! Becker.
//!
//! @see https://www.kalmanfilter.net/multiExamples.html#ex10
//!
//! @details In this example, we will estimate the altitude of a rocket. The
//! rocket is equipped with an onboard altimeter that provides altitude
//! measurements. In addition to an altimeter, the rocket is equipped with an
//! accelerometer that measures the rocket acceleration. The accelerometer
//! serves as a control input to the Kalman Filter. We assume constant
//! acceleration dynamics. Accelerometers don't sense gravity. An accelerometer
//! at rest on a table would measure 1g upwards, while accelerometers in free
//! fall will measure zero. Thus, we need to subtract the gravitational
//! acceleration constant g from each accelerometer measurement. The
//! accelerometer measurement at time step n is: an = a − g + ϵ where:
//! - a is the actual acceleration of the object (the second derivative of the
//! object position).
//! - g is the gravitational acceleration constant; g = -9.8 m.s^-2.
//! - ϵ is the accelerometer measurement error.
//! In this example, we have a control variable u, which is based on the
//! accelerometer measurement. The system state xn is defined by: xn = [ pn vn ]
//! where: pn is the rocket altitude at time n. vn is the rocket velocity at
//! time n.
//! Let us assume a rocket boosting vertically with constant acceleration. The
//! rocket is equipped with an altimeter that provides altitude measurements and
//! an accelerometer that serves as a control input.
//! - The measurements period: Δt = 0.25s
//! - The rocket acceleration: a= 30 m.s^-2
//! - The altimeter measurement error standard deviation: σxm = 20m
//! - The accelerometer measurement error standard deviation: ϵ = 0.1 m.s^-2
//!
//! @example rocket_altitude.cpp
[[maybe_unused]] auto rocket_altitude{ [] {
  // A 2x1x1 filter, constant acceleration dynamic model, no control, step time.
  using kalman = kalman<vector<double, 2>, double, double, std::tuple<>,
                        std::tuple<std::chrono::milliseconds>>;
  kalman k;

  // Initialization
  // We don't know the rocket location; we will set initial position and
  // velocity to 0.
  k.x(0., 0.);

  // Since our initial state vector is a guess, we will set a very high estimate
  // uncertainty. The high estimate uncertainty results in high Kalman gain,
  // giving a high weight to the measurement.
  k.p(kalman::estimate_uncertainty{ { 500, 0 }, { 0, 500 } });

  // Prediction
  // We will assume a discrete noise model - the noise is different at each time
  // period, but it is constant between time periods. In our previous example,
  // we used the system's random variance in acceleration σ^2 as a multiplier of
  // the process noise matrix. But here, we have an accelerometer that measures
  // the system random acceleration. The accelerometer error v is much lower
  // than system's random acceleration, therefore we use ϵ^2 as a multiplier of
  // the process noise matrix. This makes our estimation uncertainty much lower!
  k.q([](const kalman::state &x, const std::chrono::milliseconds &delta_time) {
    static_cast<void>(x);
    const auto dt{ std::chrono::duration<double>(delta_time).count() };
    return kalman::process_uncertainty{
      { 0.1 * 0.1 * dt * dt * dt * dt / 4, 0.1 * 0.1 * dt * dt * dt / 2 },
      { 0.1 * 0.1 * dt * dt * dt / 2, 0.1 * 0.1 * dt * dt }
    };
  });

  // The state transition matrix F would be:
  k.f([](const kalman::state &x, const std::chrono::milliseconds &delta_time,
         const kalman::input &u) {
    static_cast<void>(x);
    static_cast<void>(u);
    const auto dt{ std::chrono::duration<double>(delta_time).count() };
    return kalman::state_transition{ { 1, dt }, { 0, 1 } };
  });

  // The control matrix G would be:
  k.g([](const std::chrono::milliseconds &delta_time) {
    const auto dt{ std::chrono::duration<double>(delta_time).count() };
    return kalman::input_control{ 0.0313, dt };
  });

  // We also don't know what the rocket acceleration is, but we can assume that
  // it's greater than zero. Let's assume: u0 = g
  const double gravity{ -9.8 }; // [m.s^-2]
  const std::chrono::milliseconds delta_time{ 250 };
  k.predict(delta_time, -gravity);

  assert(0.3 - 0.1 < k.x()(0) && k.x()(0) < 0.3 + 0.1 &&
         2.45 - 0.1 < k.x()(1) && k.x()(1) < 2.45 + 0.1);
  assert(531.25 - 0.1 < k.p()(0, 0) && k.p()(0, 0) < 531.25 + 0.1 &&
         125 - 0.1 < k.p()(0, 1) && k.p()(0, 1) < 125 + 0.1 &&
         125 - 0.1 < k.p()(1, 0) && k.p()(1, 0) < 125 + 0.1 &&
         500 - 0.1 < k.p()(1, 1) && k.p()(1, 1) < 500 + 0.1);

  // Measure and Update
  // The dimension of zn is 1x1 and the dimension of xn is 2x1, so the dimension
  // of the observation matrix H will be 1x2.
  k.h(kalman::output_model{ 1., 0. });

  // For the sake of the example simplicity, we will assume a constant
  // measurement uncertainty: R1 = R2...Rn-1 = Rn = R.
  k.r(kalman::output_uncertainty{ 400. });

  k.update(-32.4);

  assert(-18.35 - 0.1 < k.x()(0) && k.x()(0) < -18.35 + 0.1 &&
         -1.94 - 0.1 < k.x()(1) && k.x()(1) < -1.94 + 0.1);
  assert(228.2 - 0.1 < k.p()(0, 0) && k.p()(0, 0) < 228.2 + 0.1 &&
         53.7 - 0.1 < k.p()(0, 1) && k.p()(0, 1) < 53.7 + 0.1 &&
         53.7 - 0.1 < k.p()(1, 0) && k.p()(1, 0) < 53.7 + 0.1 &&
         483.2 - 0.1 < k.p()(1, 1) && k.p()(1, 1) < 483.2 + 0.1);

  k.predict(delta_time, 39.72 + gravity);

  assert(-17.9 - 0.1 < k.x()(0) && k.x()(0) < -17.9 + 0.1 &&
         5.54 - 0.1 < k.x()(1) && k.x()(1) < 5.54 + 0.1);
  assert(285.2 - 0.1 < k.p()(0, 0) && k.p()(0, 0) < 285.2 + 0.1 &&
         174.5 - 0.1 < k.p()(0, 1) && k.p()(0, 1) < 174.5 + 0.1 &&
         174.5 - 0.1 < k.p()(1, 0) && k.p()(1, 0) < 174.5 + 0.1 &&
         483.2 - 0.1 < k.p()(1, 1) && k.p()(1, 1) < 483.2 + 0.1);

  // And so on, run a step of the filter, updating and predicting, every
  // measurements period: Δt = 250ms. The period is constant but passed as
  // variable for the example. The lambda helper shows how to simplify the
  // filter step call.
  const auto step{ [&k](const auto &...args) {
    k.template operator()<double>(args...);
  } };

  step(delta_time, 40.02 + gravity, -11.1);

  assert(-12.3 - 0.1 < k.x()(0) && k.x()(0) < -12.3 + 0.1 &&
         14.8 - 0.1 < k.x()(1) && k.x()(1) < 14.8 + 0.1);
  assert(244.9 - 0.1 < k.p()(0, 0) && k.p()(0, 0) < 244.9 + 0.1 &&
         211.6 - 0.1 < k.p()(0, 1) && k.p()(0, 1) < 211.6 + 0.1 &&
         211.6 - 0.1 < k.p()(1, 0) && k.p()(1, 0) < 211.6 + 0.1 &&
         438.8 - 0.1 < k.p()(1, 1) && k.p()(1, 1) < 438.8 + 0.1);

  step(delta_time, 39.97 + gravity, 18.);
  step(delta_time, 39.81 + gravity, 22.9);
  step(delta_time, 39.75 + gravity, 19.5);
  step(delta_time, 39.6 + gravity, 28.5);
  step(delta_time, 39.77 + gravity, 46.5);
  step(delta_time, 39.83 + gravity, 68.9);
  step(delta_time, 39.73 + gravity, 48.2);
  step(delta_time, 39.87 + gravity, 56.1);
  step(delta_time, 39.81 + gravity, 90.5);
  step(delta_time, 39.92 + gravity, 104.9);
  step(delta_time, 39.78 + gravity, 140.9);
  step(delta_time, 39.98 + gravity, 148.);
  step(delta_time, 39.76 + gravity, 187.6);
  step(delta_time, 39.86 + gravity, 209.2);
  step(delta_time, 39.61 + gravity, 244.6);
  step(delta_time, 39.86 + gravity, 276.4);
  step(delta_time, 39.74 + gravity, 323.5);
  step(delta_time, 39.87 + gravity, 357.3);
  step(delta_time, 39.63 + gravity, 357.4);
  step(delta_time, 39.67 + gravity, 398.3);
  step(delta_time, 39.96 + gravity, 446.7);
  step(delta_time, 39.8 + gravity, 465.1);
  step(delta_time, 39.89 + gravity, 529.4);
  step(delta_time, 39.85 + gravity, 570.4);
  step(delta_time, 39.9 + gravity, 636.8);
  step(delta_time, 39.81 + gravity, 693.3);
  step(delta_time, 39.81 + gravity, 707.3);

  k.update(748.5);

  // The Kalman gain for altitude converged to 0.12, which means that the
  // estimation weight is much higher than the measurement weight.
  assert(49.3 - 0.1 < k.p()(0, 0) && k.p()(0, 0) < 49.3 + 0.1 &&
         "At this point, the altitude uncertainty px = 49.3, which means that "
         "the standard deviation of the prediction is square root of 49.3: "
         "7.02m (remember that the standard deviation of the measurement is "
         "20m).");

  k.predict(delta_time, 39.68 + gravity);

  // At the beginning, the estimated altitude is influenced by measurements and
  // it is not aligned well with the true rocket altitude, since the
  // measurements are very noisy. But as the Kalman gain converges, the noisy
  // measurement has less influence and the estimated altitude is well aligned
  // with the true altitude. In this example we don't have any maneuvers that
  // cause acceleration changes, but if we had, the control input
  // (accelerometer) would update the state extrapolation equation.
  assert(831.5 - 0.1 < k.x()(0) && k.x()(0) < 831.5 + 0.1 &&
         222.94 - 0.1 < k.x()(1) && k.x()(1) < 222.94 + 0.1);
  assert(54.3 - 0.1 < k.p()(0, 0) && k.p()(0, 0) < 54.3 + 0.1 &&
         10.4 - 0.1 < k.p()(0, 1) && k.p()(0, 1) < 10.4 + 0.1 &&
         10.4 - 0.1 < k.p()(1, 0) && k.p()(1, 0) < 10.4 + 0.1 &&
         2.6 - 0.1 < k.p()(1, 1) && k.p()(1, 1) < 2.6 + 0.1);

  return 0;
}() };

} // namespace
} // namespace fcarouge::eigen::sample
